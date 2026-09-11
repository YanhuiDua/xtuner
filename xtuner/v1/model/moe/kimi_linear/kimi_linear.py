# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Linear text tower (Kimi-K3 language model) for XTuner V1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from pydantic import Field, computed_field
from typing_extensions import Self, override

from xtuner.v1.model.moe.moe import BalancingLossConfig, MoE, MoEConfig
from xtuner.v1.module.attention import KDAConfig, MLAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.linear import build_linear

from .decoder_layer import KimiDenseDecoderLayer, KimiMoEDecoderLayer


class KimiLinear(MoE):
    dense_decoder_layer_cls = KimiDenseDecoderLayer
    moe_decoder_layer_cls = KimiMoEDecoderLayer

    def __init__(self, config: "KimiLinearConfig"):
        super().__init__(config)
        self.config: KimiLinearConfig
        if config.attn_res_block_size is not None:
            self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = build_linear(config.hidden_size, 1, bias=False)

    @override
    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        assert isinstance(config, KimiLinearConfig)
        layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            if config.layers_type[layer_idx] in ["full_attention", "sliding_attention"]:
                attention_config = config.attention
            elif config.layers_type[layer_idx] == "linear_attention":
                assert config.linear_attention is not None
                attention_config = config.linear_attention
            else:
                raise ValueError(f"Unsupported layer type {config.layers_type[layer_idx]}")

            common_kimi = dict(
                situ_beta=config.activation_situ_beta or 1.0,
                situ_linear_beta=config.activation_situ_linear_beta,
                attn_res_block_size=config.attn_res_block_size,
            )
            if layer_idx < config.first_k_dense_replace:
                layers[str(layer_idx)] = self.dense_decoder_layer_cls(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    mlp_bias=config.mlp_bias,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=config.rms_norm_eps,
                    rms_norm_type=config.rms_norm_type,
                    attention_config=attention_config,
                    layer_type=config.layers_type[layer_idx],
                    rope_scaling_cfg=config.rope_scaling_cfg,
                    generate_config=config.generate_config,
                    float8_cfg=config.float8_cfg,
                    layer_idx=layer_idx,
                    **common_kimi,
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
                    layer_type=config.layers_type[layer_idx],
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
                    **common_kimi,
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
    # Hybrid attention schedule from HF linear_attn_config
    kda_layers: list[int] = Field(default_factory=list)
    full_attn_layers: list[int] = Field(default_factory=list)
    moe_act_fn_cfg: MoEActFnConfig = MoEActFnConfig(
        act_type="situ", situ_beta=4.0, situ_linear_beta=25.0
    )

    def build(self) -> KimiLinear:
        return KimiLinear(self)

    @computed_field  # type: ignore[prop-decorator]
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        types: list[Literal["full_attention", "linear_attention"]] = []
        for i in range(self.num_hidden_layers):
            # HF uses 1-based layer indices in kda_layers / full_attn_layers
            if (i + 1) in self.kda_layers:
                types.append("linear_attention")
            else:
                types.append("full_attention")
        return types

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
        kda_layers = list(linear_cfg.get("kda_layers", []))
        full_attn_layers = list(linear_cfg.get("full_attn_layers", []))

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

        return cls(
            vocab_size=text.vocab_size,
            max_position_embeddings=getattr(text, "max_position_embeddings", 4096),
            pad_token_id=getattr(text, "pad_token_id", None),
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
            kda_layers=kda_layers,
            full_attn_layers=full_attn_layers,
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

    @property
    def hf_config(self):
        # trust_remote_code model — no built-in HF config class
        return None


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
    first_k_dense_replace: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    routed_expert_hidden_size: int | None = 128
    latent_moe_use_norm: bool = True
    attn_res_block_size: int | None = 2
    kda_layers: list[int] = Field(default_factory=lambda: [1, 2, 3])
    full_attn_layers: list[int] = Field(default_factory=lambda: [4])
    attention: MLAConfig = MLAConfig(
        num_attention_heads=4,
        head_dim=64,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=48,
        v_head_dim=48,
        use_output_gate=True,
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
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=10000.0)
    )
    balancing_loss_cfg: BalancingLossConfig | None = None
    tie_word_embeddings: bool = False


class KimiLinearK3TextConfig(KimiLinearConfig):
    """Published Kimi-K3 text-tower dimensions (structure only; weights separate)."""

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
    # Subset listed in Hub config; full lists applied via from_hf
    kda_layers: list[int] = Field(default_factory=list)
    full_attn_layers: list[int] = Field(default_factory=list)
    attention: MLAConfig = MLAConfig(
        num_attention_heads=96,
        head_dim=192,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        use_output_gate=True,
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
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=10000.0)
    )
