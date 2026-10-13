# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Linear text tower (Kimi-K3 language model) for XTuner V1."""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from shutil import copy
from typing import Callable, Literal, cast

import torch
import torch.nn as nn
from pydantic import Field, PrivateAttr, computed_field
from typing_extensions import Self, override

from transformers.configuration_utils import PretrainedConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoE, MoEConfig
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.attention import KDAConfig, MLAConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import (
    DenseDecoderLayerMicroBatchOutput,
    DenseDecoderLayerOutput,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .decoder_layer import (
    KimiDenseDecoderLayer,
    KimiMoEDecoderLayer,
    apply_attn_res,
)


def _kimi_hybrid_layer_ids(num_hidden_layers: int) -> tuple[list[int], list[int]]:
    """Hub schedule (1-based): 3×KDA + 1×MLA repeating; last layer always
    MLA."""
    kda = [i for i in range(1, num_hidden_layers + 1) if i % 4 != 0 and i != num_hidden_layers]
    full = [i for i in range(1, num_hidden_layers + 1) if i not in set(kda)]
    return kda, full


class KimiLinear(MoE):
    dense_decoder_layer_cls = KimiDenseDecoderLayer
    moe_decoder_layer_cls = KimiMoEDecoderLayer

    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.config: KimiLinearConfig

    @override
    def _configure_model_specific_layers(self) -> None:
        if self.config.attn_res_block_size is not None:
            self.output_attn_res_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.output_attn_res_proj = cast(nn.Linear, build_linear(self.config.hidden_size, 1, bias=False))

    def _apply_output_attn_res(self, hidden_states: torch.Tensor, block_residual: torch.Tensor | None) -> torch.Tensor:
        if block_residual is None or self.config.attn_res_block_size is None:
            return hidden_states
        batch_size, seq_len, hidden_size = hidden_states.shape
        return apply_attn_res(
            hidden_states.view(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch_size, seq_len, hidden_size)

    def _seed_block_residual(
        self, hidden_states: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        """HF AttnRes empty buffer: ``(T, 0, H)`` per micro-batch (``T = B * S``)."""
        if isinstance(hidden_states, list):
            return [cast(torch.Tensor, self._seed_block_residual(hs)) for hs in hidden_states]
        b, s, h = hidden_states.shape
        return hidden_states.new_zeros(b * s, 0, h)

    @override
    def _call_decoder_layer(
        self,
        *,
        decoder_layer: torch.nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        previous_layer_results: (
            DenseDecoderLayerOutput
            | DenseDecoderLayerMicroBatchOutput
            | MoEDecoderLayerOutput
            | MoEDecoderLayerMicroBatchOutput
            | torch.Tensor
            | tuple[torch.Tensor, ...]
            | None
        ),
    ) -> (
        DenseDecoderLayerOutput
        | DenseDecoderLayerMicroBatchOutput
        | MoEDecoderLayerOutput
        | MoEDecoderLayerMicroBatchOutput
    ):
        if (
            previous_layer_results is not None
            and not isinstance(previous_layer_results, (torch.Tensor, tuple))
            and "block_residual" in previous_layer_results
        ):
            block_residual = previous_layer_results["block_residual"]  # type: ignore[typeddict-item]
        elif self.config.attn_res_block_size is not None:
            block_residual = self._seed_block_residual(hidden_states)
        else:
            block_residual = None

        offload_tensors: list[torch.Tensor] = []
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1 and layer_idx >= self.config.first_k_dense_replace:
            offload_tensors = list(hidden_states) if isinstance(hidden_states, list) else [hidden_states]

        offload_block_idx = sum(
            self.config.first_k_dense_replace <= int(previous_idx) < layer_idx for previous_idx in self.layers
        )
        with self._saved_tensors_offload_ctx(offload_block_idx, offload_tensors):
            return cast(Callable[..., object], decoder_layer)(  # type: ignore[return-value]
                hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                block_residual=block_residual,
            )

    @override
    def _post_decoder_stack(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        previous_layer_results: (
            DenseDecoderLayerOutput
            | DenseDecoderLayerMicroBatchOutput
            | MoEDecoderLayerOutput
            | MoEDecoderLayerMicroBatchOutput
            | None
        ),
    ) -> torch.Tensor | list[torch.Tensor]:
        block_residual = None
        if previous_layer_results is not None and "block_residual" in previous_layer_results:
            block_residual = previous_layer_results["block_residual"]  # type: ignore[typeddict-item]

        if isinstance(hidden_states, list):
            br_list: list[torch.Tensor | None]
            if block_residual is None:
                br_list = [None] * len(hidden_states)
            elif isinstance(block_residual, list):
                br_list = list(block_residual)
            else:
                br_list = [block_residual] * len(hidden_states)
            return [self._apply_output_attn_res(hs, br) for hs, br in zip(hidden_states, br_list)]

        assert block_residual is None or isinstance(block_residual, torch.Tensor)
        return self._apply_output_attn_res(hidden_states, block_residual)

    @override
    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        assert isinstance(config, KimiLinearConfig)
        assert isinstance(config.attention, MLAConfig)
        layers = nn.ModuleDict()
        layer_types = cast(
            list[Literal["full_attention", "sliding_attention", "linear_attention"]],
            config.layers_type,
        )
        attention_config: MLAConfig | KDAConfig
        for layer_idx in range(config.num_hidden_layers):
            if layer_types[layer_idx] in ["full_attention", "sliding_attention"]:
                attention_config = config.attention
            elif layer_types[layer_idx] == "linear_attention":
                assert isinstance(config.linear_attention, KDAConfig)
                attention_config = config.linear_attention
            else:
                raise ValueError(f"Unsupported layer type {layer_types[layer_idx]}")

            situ_beta = config.activation_situ_beta or 1.0
            situ_linear_beta = config.activation_situ_linear_beta
            attn_res_block_size = config.attn_res_block_size
            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = self.dense_decoder_layer_cls(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    attention_config=attention_config,
                    layer_type=layer_types[layer_idx],
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    attn_res_block_size=attn_res_block_size,
                )
            else:
                layers[str(layer_idx)] = self.moe_decoder_layer_cls(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    moe_intermediate_size=config.moe_intermediate_size,
                    mlp_bias=config.mlp_bias,
                    gate_bias=config.gate_bias,
                    moe_bias=config.moe_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    num_experts_per_tok=config.num_experts_per_tok,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    with_shared_expert_gate=config.with_shared_expert_gate,
                    hidden_factor=config.hidden_factor,
                    layer_type=layer_types[layer_idx],
                    attention_config=attention_config,
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    router_config=config.router,
                    router_compute_dtype=config.router_compute_dtype,
                    moe_act_fn_cfg=config.moe_act_fn_cfg,
                    expert_backend=config.expert_backend,
                    sonicmoe_cfg=config.sonicmoe_cfg,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    dispatcher=config.dispatcher,
                    ep_mesh=self.ep_mesh,
                    expert_tp_mesh=self.expert_tp_mesh,
                    ep_tp_mesh=self.ep_tp_mesh,
                    routed_expert_hidden_size=config.routed_expert_hidden_size,
                    latent_moe_use_norm=config.latent_moe_use_norm,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    attn_res_block_size=attn_res_block_size,
                )
        from xtuner.v1.model.utils import module_dict_repr

        layers.__class__.__repr__ = module_dict_repr  # type: ignore[method-assign]
        return layers

    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        # XTuner MoE flat names -> HF block_sparse_moe
        if "layers" in key:
            key = re.sub(
                r"layers\.(\d+)\.(experts|gate|shared_experts|routed_expert_down_proj|"
                r"routed_expert_up_proj|routed_expert_norm)",
                r"layers.\1.block_sparse_moe.\2",
                key,
            )

        n_routed_experts = self.config.n_routed_experts
        if "fused_w1w3.weight" in key:
            out: list[str] = []
            for i in range(n_routed_experts):
                out.append(key.replace("fused_w1w3.weight", f"{i}.w1.weight"))
                out.append(key.replace("fused_w1w3.weight", f"{i}.w3.weight"))
            return out
        if "fused_w2.weight" in key:
            out = []
            for i in range(n_routed_experts):
                out.append(key.replace("fused_w2.weight", f"{i}.w2.weight"))
            return out

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        if key.startswith("output_attn_res_"):
            return [f"model.{key}"]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]


class KimiLinearConfig(MoEConfig):
    """Base config for Kimi Linear (`model_type=kimi_linear`)."""

    model_type: str | None = "kimi_linear"
    hidden_act: str = "situ"
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = False
    attn_res_block_size: int | None = None
    bos_token_id: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_nextn_predict_layers: int = 0
    hf_head_dim: int | None = None
    hf_num_key_value_heads: int | None = None
    hf_architectures: list[str] = Field(default_factory=lambda: ["KimiLinearForCausalLM"])
    hf_auto_map: dict[str, str] = Field(
        default_factory=lambda: {
            "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
            "AutoModel": "modeling_kimi_linear.KimiLinearModel",
            "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM",
        }
    )
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="situ", situ_beta=4.0, situ_linear_beta=25.0)
    _hf_config_cls: Callable[..., PretrainedConfig] | None = PrivateAttr(default=None)
    _hf_config_source_file: Path | None = PrivateAttr(default=None)

    def build(self) -> KimiLinear:
        return KimiLinear(self)

    @computed_field  # type: ignore[prop-decorator]
    def layers_type(self) -> list[Literal["full_attention", "sliding_attention", "linear_attention"]]:
        # Hub Kimi-K3: full attn when 1-based index % 4 == 0, or the final layer.
        return [
            ("full_attention" if (i + 1) % 4 == 0 or (i + 1) == self.num_hidden_layers else "linear_attention")
            for i in range(self.num_hidden_layers)
        ]

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        # Accept either standalone kimi_linear or kimi_k3 with text_config
        if getattr(hf_config, "model_type", None) == "kimi_k3":
            text = hf_config.text_config
        else:
            assert getattr(hf_config, "model_type", None) == "kimi_linear", (
                f"Expected kimi_linear or kimi_k3, got {getattr(hf_config, 'model_type', None)}"
            )
            text = hf_config

        linear_cfg = getattr(text, "linear_attn_config", None) or {}

        rope_parameters_cfg = RopeParametersConfig.from_hf_config(text)
        attention = MLAConfig(
            num_attention_heads=text.num_attention_heads,
            head_dim=(text.qk_nope_head_dim or 0) + (text.qk_rope_head_dim or 0),
            kv_lora_rank=text.kv_lora_rank,
            q_lora_rank=text.q_lora_rank,
            qk_rope_head_dim=text.qk_rope_head_dim,
            qk_nope_head_dim=text.qk_nope_head_dim,
            v_head_dim=text.v_head_dim,
            use_output_gate=bool(getattr(text, "mla_use_output_gate", False)),
            use_nope=bool(getattr(text, "mla_use_nope", False)),
        )
        linear_attention = KDAConfig(
            num_heads=linear_cfg.get("num_heads", text.num_attention_heads),
            head_dim=linear_cfg.get("head_dim", 128),
            conv_kernel_size=linear_cfg.get("short_conv_kernel_size", 4),
            use_full_rank_gate=bool(linear_cfg.get("use_full_rank_gate", True)),
            gate_lower_bound=linear_cfg.get("gate_lower_bound", -5.0),
            rms_norm_eps=text.rms_norm_eps,
        )
        situ_beta = getattr(text, "activation_situ_beta", 4.0) or 4.0
        situ_linear_beta = getattr(text, "activation_situ_linear_beta", 25.0)

        config = cls(
            vocab_size=text.vocab_size,
            max_position_embeddings=getattr(text, "max_position_embeddings", 4096),
            pad_token_id=getattr(text, "pad_token_id", None),
            bos_token_id=getattr(text, "bos_token_id", None),
            eos_token_id=text.eos_token_id,
            num_hidden_layers=text.num_hidden_layers,
            hidden_size=text.hidden_size,
            intermediate_size=text.intermediate_size,
            moe_intermediate_size=text.moe_intermediate_size,
            rms_norm_eps=text.rms_norm_eps,
            hidden_act=text.hidden_act,
            rope_parameters_cfg=rope_parameters_cfg,
            attention=attention,
            linear_attention=linear_attention,
            n_routed_experts=text.num_experts,
            n_shared_experts=text.num_shared_experts or 0,
            num_experts_per_tok=text.num_experts_per_token,
            first_k_dense_replace=text.first_k_dense_replace,
            routed_expert_hidden_size=getattr(text, "routed_expert_hidden_size", None),
            latent_moe_use_norm=bool(getattr(text, "latent_moe_use_norm", False)),
            attn_res_block_size=getattr(text, "attn_res_block_size", None),
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            initializer_range=getattr(text, "initializer_range", 0.02),
            use_cache=bool(getattr(text, "use_cache", True)),
            moe_layer_freq=getattr(text, "moe_layer_freq", 1),
            use_grouped_topk=bool(getattr(text, "use_grouped_topk", True)),
            num_nextn_predict_layers=getattr(text, "num_nextn_predict_layers", 0),
            hf_head_dim=getattr(text, "head_dim", None),
            hf_num_key_value_heads=getattr(text, "num_key_value_heads", None),
            hf_architectures=list(getattr(text, "architectures", None) or ["KimiLinearForCausalLM"]),
            hf_auto_map=dict(
                getattr(text, "auto_map", None)
                or {
                    "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
                    "AutoModel": "modeling_kimi_linear.KimiLinearModel",
                    "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM",
                }
            ),
            router=NoAuxRouterConfig(
                scoring_func="sigmoid" if text.moe_router_activation_func == "sigmoid" else "softmax",
                router_scaling_factor=text.routed_scaling_factor,
                norm_topk_prob=text.moe_renormalize,
                n_group=text.num_expert_group,
                topk_group=text.topk_group,
            ),
            moe_act_fn_cfg=MoEActFnConfig(
                act_type="situ" if text.hidden_act == "situ" else "swiglu",
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            ),
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=None,
            tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        )
        # Kimi-K3 is a trust_remote_code model in the pinned Transformers
        # version. Reuse its official config class instead of serializing a
        # generic PretrainedConfig (which writes an empty model_type).
        config._hf_config_cls = type(text)
        source_file = inspect.getsourcefile(type(text))
        config._hf_config_source_file = Path(source_file) if source_file is not None else None
        return config

    def save_hf(self, hf_path: str | Path):
        """Save config JSON together with its trust-remote-code config
        module."""
        super().save_hf(hf_path)
        if self._hf_config_source_file is None or not self._hf_config_source_file.is_file():
            return

        auto_config = self.hf_auto_map.get("AutoConfig")
        if auto_config is None:
            return
        module_name = auto_config.split(".", 1)[0]
        target = Path(hf_path) / f"{module_name}.py"
        if not target.exists():
            copy(self._hf_config_source_file, target)

    @property
    def hf_config(self) -> PretrainedConfig | None:
        """Build the standalone text config with the official remote class."""
        if self._hf_config_cls is None:
            return None
        assert isinstance(self.router, NoAuxRouterConfig), "Only support saving NoAuxRouter to HF KimiLinear format."
        attention = self.attention
        assert isinstance(attention, MLAConfig)
        linear_attn = self.linear_attention
        assert linear_attn is None or isinstance(linear_attn, KDAConfig)
        kda_layers, full_attn_layers = _kimi_hybrid_layer_ids(self.num_hidden_layers)
        rope_theta = self.rope_parameters_cfg.rope_theta if self.rope_parameters_cfg is not None else 10000.0
        return self._hf_config_cls(
            model_type="kimi_linear",
            architectures=self.hf_architectures,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            head_dim=self.hf_head_dim,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=attention.num_attention_heads,
            num_key_value_heads=self.hf_num_key_value_heads or attention.num_attention_heads,
            num_experts=self.n_routed_experts,
            num_experts_per_token=self.num_experts_per_tok,
            num_shared_experts=self.n_shared_experts,
            first_k_dense_replace=self.first_k_dense_replace,
            hidden_act=self.hidden_act,
            activation_situ_beta=self.activation_situ_beta,
            activation_situ_linear_beta=self.activation_situ_linear_beta,
            routed_expert_hidden_size=self.routed_expert_hidden_size,
            latent_moe_use_norm=self.latent_moe_use_norm,
            attn_res_block_size=self.attn_res_block_size,
            rms_norm_eps=self.rms_norm_eps,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=rope_theta,
            q_lora_rank=attention.q_lora_rank,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            mla_use_nope=attention.use_nope,
            mla_use_output_gate=attention.use_output_gate,
            moe_router_activation_func="sigmoid" if self.router.scoring_func == "sigmoid" else "softmax",
            moe_renormalize=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            moe_layer_freq=self.moe_layer_freq,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.router.n_group,
            topk_group=self.router.topk_group,
            topk_method="noaux_tc",
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            linear_attn_config={
                "kda_layers": kda_layers,
                "full_attn_layers": full_attn_layers,
                "num_heads": linear_attn.num_heads if linear_attn else attention.num_attention_heads,
                "head_dim": linear_attn.head_dim if linear_attn else 128,
                "short_conv_kernel_size": linear_attn.conv_kernel_size if linear_attn else 4,
                "use_full_rank_gate": bool(linear_attn.use_full_rank_gate) if linear_attn else True,
                "gate_lower_bound": linear_attn.gate_lower_bound if linear_attn else -5.0,
            },
            auto_map=self.hf_auto_map,
            dtype="bfloat16",
        )


# 虚拟的小尺寸模型
class KimiLinearTinyConfig(KimiLinearConfig):
    """Tiny random-weight config for load / smoke tests."""

    vocab_size: int = 1024
    max_position_embeddings: int = 2048
    pad_token_id: int | None = 0
    eos_token_id: int = 2
    num_hidden_layers: int = 4
    hidden_size: int = 256
    intermediate_size: int = 512
    moe_intermediate_size: int = 128
    rms_norm_eps: float = 1e-5
    hidden_act: str = "situ"
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    use_grouped_topk: bool = False
    first_k_dense_replace: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    routed_expert_hidden_size: int | None = 128
    latent_moe_use_norm: bool = True
    attn_res_block_size: int | None = 2
    attention: MLAConfig = MLAConfig(
        num_attention_heads=4,
        head_dim=64,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=48,
        v_head_dim=48,
        use_output_gate=True,
        use_nope=True,
    )
    linear_attention: KDAConfig | None = KDAConfig(
        num_heads=4,
        head_dim=32,
        conv_kernel_size=4,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
    )
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="situ", situ_beta=4.0, situ_linear_beta=25.0)
    rope_parameters_cfg: RopeParametersConfig = Field(default_factory=lambda: RopeParametersConfig(rope_theta=10000.0))
    balancing_loss_cfg: BalancingLossConfig | None = None
    tie_word_embeddings: bool = False


# text tower的配置（完整模型2.8T）
class KimiLinearK3TextConfig(KimiLinearConfig):
    """Published Kimi-K3 text-tower dimensions (structure only; weights
    separate)."""

    vocab_size: int = 163840
    max_position_embeddings: int = 1048576
    pad_token_id: int | None = 163839
    eos_token_id: int = 163586
    num_hidden_layers: int = 93
    hidden_size: int = 7168
    intermediate_size: int = 33792
    moe_intermediate_size: int = 3072
    rms_norm_eps: float = 1e-5
    first_k_dense_replace: int = 1
    n_routed_experts: int = 896
    n_shared_experts: int = 2
    num_experts_per_tok: int = 16
    routed_expert_hidden_size: int | None = 3584
    latent_moe_use_norm: bool = True
    attn_res_block_size: int | None = 12
    attention: MLAConfig = MLAConfig(
        num_attention_heads=96,
        head_dim=192,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        use_output_gate=True,
        use_nope=True,
    )
    linear_attention: KDAConfig | None = KDAConfig(
        num_heads=96,
        head_dim=128,
        conv_kernel_size=4,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
    )
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(act_type="situ", situ_beta=4.0, situ_linear_beta=25.0)
    rope_parameters_cfg: RopeParametersConfig = Field(default_factory=lambda: RopeParametersConfig(rope_theta=10000.0))
