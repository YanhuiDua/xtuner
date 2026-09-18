# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 ConceptLM: encoder / predictor / decoder composition with MUDD.

Port of the reference implementation (``megatron_ncp_qwen35`` @
``jxxiong/residual``, ``examples/conceptlm``) onto XTuner's Qwen3.5 dense
tower:

    tokens -> encoder stack (optional MUDD)
           -> chunk-4 mean-pooled concepts
           -> predictor stack (optional MUDD) + 32x128 residual-VQ heads
           -> HLM objective (each concept predicts the next one)
           -> decoder stack (optional MUDD) fed with encoder hidden states
              plus the expanded concept features
           -> final norm + LM head

All three stacks are ordinary Qwen3.5 hybrid layers (3 GatedDeltaNet layers
then one full-attention layer), so layer construction mirrors
:class:`xtuner.v1.model.dense.Dense` and MUDD is wired per stack the same way
``Qwen3VLTextDense.forward`` wires it for the 9B tower.
"""

import os
from typing import Literal, Self

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import computed_field
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from tqdm import tqdm
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.module import LMHead, RMSNorm
from xtuner.v1.module.concept_quantizer import ConceptQuantizer
from xtuner.v1.module.concept_sequence import (
    build_concept_context,
    document_lengths,
    expand_concepts,
    merge_chunks,
    shift_concepts_left,
    successor_mask,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.mudd import Qwen35MUDD
from xtuner.v1.utils import get_device, log_rank0

from .dense import DenseModelOutputs
from .qwen3_5_text import Qwen3_5_VLTextDense2BConfig


DEVICE = get_device()


def stack_layer_types(num_layers: int) -> list[Literal["full_attention", "linear_attention"]]:
    """Qwen3.5 hybrid pattern of one stack: 3 GDN layers, then full
    attention."""
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]


class ConceptLMQwen35Config(Qwen3_5_VLTextDense2BConfig):
    """Qwen3.5-2B ConceptLM recipe (defaults follow the reference launcher)."""

    # stack depths: reference ``--conceptlm-{encoder,predictor,decoder}-layers``
    concept_encoder_layers: int = 12
    concept_predictor_layers: int = 8
    concept_decoder_layers: int = 12
    # concept transform
    concept_chunk_size: int = 4
    concept_chunk_merge_method: Literal["meanpooling", "first", "last"] = "meanpooling"
    concept_shift_feature: bool = True
    # residual VQ
    concept_codebook_size: int = 128
    concept_num_codebooks: int = 32
    concept_commitment_cost: float = 0.25
    concept_merge_mode: Literal["raw_logits", "softmax", "hard_top1"] = "raw_logits"
    # HLM objective
    concept_detach_hlm_target: bool = True
    concept_vq_loss_weight: float = 1.0
    concept_hlm_loss_weight: float = 1.0
    # per-stack MUDD (the reference turns all three on with --conceptlm-mudd-compile)
    concept_encoder_mudd: bool = False
    concept_predictor_mudd: bool = False
    concept_decoder_mudd: bool = False

    @computed_field
    @override
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        # Only consumed by helpers that ask for the base tower pattern; every
        # ConceptLM stack derives its own pattern from its own depth.
        return stack_layer_types(self.num_hidden_layers)

    def build(self) -> "ConceptLMQwen35":  # type: ignore[override]
        return ConceptLMQwen35(self)


class ConceptLMQwen35(BaseModel):
    """ConceptLM built from three Qwen3.5 hybrid stacks with optional MUDD."""

    config: ConceptLMQwen35Config

    def __init__(self, config: ConceptLMQwen35Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = self.build_rotary_embedding(config)

        self.encoder_layers = self._build_stack(config, config.concept_encoder_layers)
        self.predictor_layers = self._build_stack(config, config.concept_predictor_layers)
        self.decoder_layers = self._build_stack(config, config.concept_decoder_layers)

        self.prediction_heads = nn.ModuleList(
            [
                build_linear(
                    config.hidden_size,
                    config.concept_codebook_size,
                    bias=True,
                    float8_cfg=config.float8_cfg,
                )
                for _ in range(config.concept_num_codebooks)
            ]
        )
        self.concept_vq_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type=config.rms_norm_type)
        self.concept_vq_pred_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type=config.rms_norm_type)
        self.concept_quantizer = ConceptQuantizer(
            hidden_size=config.hidden_size,
            num_codebooks=config.concept_num_codebooks,
            codebook_size=config.concept_codebook_size,
            commitment_cost=config.concept_commitment_cost,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, type=config.rms_norm_type)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self.encoder_mudd = (
            self._build_mudd(config, config.concept_encoder_layers) if config.concept_encoder_mudd else None
        )
        self.predictor_mudd = (
            self._build_mudd(config, config.concept_predictor_layers) if config.concept_predictor_mudd else None
        )
        self.decoder_mudd = (
            self._build_mudd(config, config.concept_decoder_layers) if config.concept_decoder_mudd else None
        )

        self._init_load_spec()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _build_stack(config: ConceptLMQwen35Config, num_layers: int) -> nn.ModuleDict:
        layers = nn.ModuleDict()
        layer_types = stack_layer_types(num_layers)
        for layer_idx in range(num_layers):
            attention_config = (
                config.attention if layer_types[layer_idx] == "full_attention" else config.linear_attention
            )
            layers[str(layer_idx)] = DenseDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mlp_bias=config.mlp_bias,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                rms_norm_type=config.rms_norm_type,
                attention_config=attention_config,
                generate_config=config.generate_config,
                rope_scaling_cfg=config.rope_scaling_cfg,
                float8_cfg=config.float8_cfg,
                # ``DenseDecoderLayer`` annotates only full/sliding attention even though
                # the Qwen3.5 towers pass ``linear_attention`` for GDN layers (same as
                # ``Dense.build_layers``); the runtime value is the correct one.
                layer_type=layer_types[layer_idx],  # type: ignore[arg-type]
                layer_idx=layer_idx,
            )
        return layers

    @staticmethod
    def _build_mudd(config: ConceptLMQwen35Config, num_layers: int) -> Qwen35MUDD:
        return Qwen35MUDD(
            config.hidden_size,
            num_layers,
            config.rms_norm_eps,
            compile_route=config.mudd_compile,
            route_kernel=config.mudd_route_kernel,
            router_norm_affine=config.mudd_router_norm_affine,
        )

    # --------------------------------------------------------------- forward
    def to_hf_key_list(self, key: str) -> list[str]:
        # Training-from-scratch model: HF checkpoint I/O is intentionally not wired
        # yet, but the load-spec bookkeeping needs a stable identity mapping.
        return [key]

    def _run_stack(
        self,
        layers: nn.ModuleDict,
        mudd: Qwen35MUDD | None,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        records: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mudd_state = mudd.initial_state(hidden_states) if mudd is not None else None
        for idx, decoder_layer in layers.items():
            current_raw = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                mudd_streams=mudd_state.streams if mudd_state is not None else None,
            )
            if records is not None:
                records.append(current_raw)
            if mudd is not None:
                assert mudd_state is not None
                mudd_state = mudd.finish_layer(int(idx), current_raw, mudd_state)
                hidden_states = mudd.final_read(mudd_state)
            else:
                hidden_states = current_raw
        return hidden_states

    def forward(  # type: ignore[override]
        self,
        seq_ctx: SequenceContext,
        loss_ctx: dict[str, BaseLossContext | list[BaseLossContext]] | None = None,
        return_intermediates: bool = False,
    ) -> DenseModelOutputs | tuple[DenseModelOutputs, dict[str, torch.Tensor]]:
        config = self.config
        input_ids = seq_ctx.input_ids
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds
        assert seq_ctx.position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids)
        self._mark_dynamic(seq_ctx)

        doc_lengths = document_lengths(seq_ctx)
        chunk_size = config.concept_chunk_size

        encoder_hidden = self._run_stack(
            self.encoder_layers, self.encoder_mudd, hidden_states, position_embeddings, seq_ctx
        )
        concept_hidden, concept_lengths = merge_chunks(
            encoder_hidden, doc_lengths, chunk_size, config.concept_chunk_merge_method
        )
        # The VQ path stops gradients into the encoder but keeps the input norm's
        # affine parameters trainable (reference keeps the same contract).
        vq_input = self.concept_vq_input_norm(concept_hidden.detach())
        vq_target = F.rms_norm(concept_hidden, (config.hidden_size,), eps=config.rms_norm_eps)
        # The quantizer follows the reference's sequence-major layout, so swap the
        # batch dimension around the call.
        quantized = self.concept_quantizer(vq_input.transpose(0, 1))

        concept_ctx = build_concept_context(concept_hidden, concept_lengths, chunk_size)
        concept_position_embeddings = self.rotary_emb(concept_hidden, concept_ctx.position_ids)
        predictor_records: list[torch.Tensor] = [] if return_intermediates else []
        predicted_hidden = self._run_stack(
            self.predictor_layers,
            self.predictor_mudd,
            concept_hidden,
            concept_position_embeddings,
            concept_ctx,
            records=predictor_records,
        )
        prediction_logits = torch.stack([head(predicted_hidden) for head in self.prediction_heads], dim=2)
        predicted = (
            self.concept_quantizer.decode_logits(prediction_logits.transpose(0, 1), mode=config.concept_merge_mode)
            .transpose(0, 1)
            .to(dtype=concept_hidden.dtype)
        )

        # HLM: predict the next concept of the same document.
        target_successor = shift_concepts_left(vq_target, concept_lengths)
        if config.concept_detach_hlm_target:
            target_successor = target_successor.detach()
        # ``successor_mask`` returns XTuner's batch-major layout, so the batch size is
        # dimension 0 (XTuner packs into a single batch element), not the concept length.
        hlm_mask = successor_mask(concept_lengths, concept_hidden.shape[0], concept_hidden.device)
        per_concept_hlm = F.mse_loss(
            self.concept_vq_pred_norm(predicted).float(), target_successor.float(), reduction="none"
        ).mean(dim=-1)
        hlm_numerator = per_concept_hlm.masked_select(hlm_mask).sum()
        hlm_count = hlm_mask.sum().to(dtype=torch.float32)

        expanded = expand_concepts(predicted, doc_lengths, concept_lengths, chunk_size, config.concept_shift_feature)
        decoder_input = encoder_hidden + expanded.to(dtype=encoder_hidden.dtype)
        decoder_hidden = self._run_stack(
            self.decoder_layers, self.decoder_mudd, decoder_input, position_embeddings, seq_ctx
        )
        hidden_states = self.norm(decoder_hidden)

        output: dict = {}
        intermediates = {
            "encoder_hidden": encoder_hidden,
            "concept_hidden": concept_hidden,
            "predicted": predicted,
            "decoder_input": decoder_input,
            "decoder_hidden": decoder_hidden,
        }
        for layer_idx, record in enumerate(predictor_records):
            intermediates[f"predictor_layer{layer_idx}"] = record
        if loss_ctx is None:
            _, (logits, _) = self.lm_head(hidden_states, None)
            output["logits"] = logits
        else:
            lm_loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx["lm"])  # type: ignore[call-overload]
            # Reference loss: sum_masked CE + chunk_size * (w_vq * vq_num + w_hlm * hlm_num),
            # reported per token, so the auxiliary terms share the LM loss scale.
            num_tokens = hidden_states.new_tensor(max(sum(doc_lengths), 1), dtype=torch.float32)
            aux_loss = chunk_size * (
                config.concept_vq_loss_weight * quantized.numerator + config.concept_hlm_loss_weight * hlm_numerator
            )
            if os.environ.get("XTUNER_CONCEPTLM_DEBUG") == "1":
                from xtuner.v1.utils import log_rank0

                log_rank0.info(
                    "[conceptlm] lm_loss=%.6f vq_num=%.6f hlm_num=%.6f n_concepts=%d "
                    "vq_count=%.3f hlm_count=%.3f num_tokens=%d loss=%.6f"
                    % (
                        float(lm_loss.detach()),
                        float(quantized.numerator.detach()),
                        float(hlm_numerator.detach()),
                        int(sum(concept_lengths)),
                        float(quantized.count.detach()),
                        float(hlm_count.detach()),
                        int(num_tokens.item()),
                        float((lm_loss + aux_loss / num_tokens).detach()),
                    )
                )
            output["loss"] = lm_loss + aux_loss / num_tokens
            output["logits"] = logits
            output["extra_info"] = extra_info
        if return_intermediates:
            return DenseModelOutputs(**output), intermediates
        return DenseModelOutputs(**output)

    # ----------------------------------------------------------- FSDP wiring
    def _stacks(self) -> list[nn.ModuleDict]:
        return [self.encoder_layers, self.predictor_layers, self.decoder_layers]

    def _init_device_mesh(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config
        world_size = dist.get_world_size()
        if self.fsdp_config.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                DEVICE,
                (world_size, 1),
                mesh_dim_names=(f"{self.config.mesh_prefix}.fsdp", f"{self.config.mesh_prefix}.others"),
            )
            self.fsdp_mesh = model_mesh[f"{self.config.mesh_prefix}.fsdp"]
        else:
            model_mesh = init_device_mesh(
                DEVICE,
                (world_size // self.fsdp_config.hsdp_sharding_size, self.fsdp_config.hsdp_sharding_size),
                mesh_dim_names=(
                    f"{self.config.mesh_prefix}.hsdp_replicate",
                    f"{self.config.mesh_prefix}.hsdp_shard",
                ),
            )
            self.hsdp_mesh = model_mesh
            self.fsdp_mesh = self.hsdp_mesh[f"{self.config.mesh_prefix}.hsdp_shard"]
        self._world_mesh = model_mesh

    @override
    def fully_shard(self, fsdp_config: FSDPConfig) -> Self:
        self.fsdp_config = fsdp_config
        self._init_device_mesh(fsdp_config)

        if self.config.float8_cfg is not None:
            from xtuner.v1.float8.float8_handler import Float8Handler

            Float8Handler.pad_for_fsdp(self, self.fsdp_mesh, callback_after_pad=self._init_load_spec)  # type: ignore[arg-type]

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        if not checkpoint_preserve_rng_state and self.config.attention.dropout > 0.0:
            checkpoint_preserve_rng_state = True
            log_rank0.warning("When using dropout, checkpoint_preserve_rng_state is set to True to avoid issues.")

        assert self.fsdp_mesh is not None
        if self.fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        setattr(module, p_name, nn.Parameter(param.to(dtype=torch.float32)))
        else:
            for param in self.parameters():
                param.requires_grad = False

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)
        lm_head_mp_policy = (
            MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            if fsdp_config.fp32_lm_head
            else mp_policy
        )
        offload_policy = CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None
        shard_mesh = self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh

        # Flatten the three stacks so the recompute budget is spread over all layers
        # (the reference applies the same concept to its composed blocks).
        stack_layers = [(stack, key, layer) for stack in self._stacks() for key, layer in stack.items()]
        num_recompute_layers = int(len(stack_layers) * self.fsdp_config.recompute_ratio)
        generator = torch.Generator()
        generator.manual_seed(0)
        shuffled = torch.randperm(len(stack_layers), generator=generator)
        for position, flat_idx in enumerate(tqdm(shuffled, desc="[FSDP Sharding]")):
            stack, key, layer = stack_layers[int(flat_idx)]
            if position < num_recompute_layers:
                layer = apply_activation_checkpointing(layer, preserve_rng_state=checkpoint_preserve_rng_state)
                if self.compile_cfg:
                    compiled_forward = torch.compile(type(layer).forward, fullgraph=False)
                    layer.forward = compiled_forward.__get__(layer, type(layer))  # type: ignore[method-assign]
            stack[key] = layer
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=layer,
            )

        # NOTE: ``prediction_heads`` is a ``nn.ModuleList`` and ``Qwen35MUDD`` has no
        # ``forward``, so neither can be an FSDP2 unit of its own; their parameters are
        # sharded together with the root module below (same as ``Dense`` does for MUDD).
        for module, policy in (
            (self.embed_tokens, mp_policy),
            (self.norm, mp_policy),
            (self.lm_head, lm_head_mp_policy),
            (self.concept_vq_input_norm, mp_policy),
            (self.concept_vq_pred_norm, mp_policy),
            (self.concept_quantizer, mp_policy),
        ):
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=module,
            )

        self._fully_shard(
            mesh=shard_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_config.reshard_after_forward,
            offload_policy=offload_policy,
        )
        self._init_load_spec()
        self._to_empty_meta()
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        return self
