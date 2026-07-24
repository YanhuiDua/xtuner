# Copyright (c) OpenMMLab. All rights reserved.

from typing import Literal

import torch
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.attention.concept_olmo import ConceptOlmoAttention, ConceptOlmoRMSNorm
from xtuner.v1.module.linear import build_linear
from xtuner.v1.ops.act_fn import get_act_fn


class ConceptOlmoMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.linear_fc1 = build_linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.linear_fc2 = build_linear(
            intermediate_size,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.linear_fc1(hidden_states).chunk(2, dim=-1)
        return self.linear_fc2(self.act_fn(gate) * value)


class ConceptOlmoDecoderLayer(nn.Module):
    """OLMo post-norm layer used in all three ConceptLM stages."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float,
        rms_norm_eps: float,
        attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"],
        sliding_window: int,
        layer_type: Literal["full_attention", "sliding_attention"],
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.self_attention = ConceptOlmoAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            attn_impl=attn_impl,
            sliding_window=sliding_window,
            layer_type=layer_type,
            float8_cfg=float8_cfg,
        )
        self.post_attention_layernorm = ConceptOlmoRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = ConceptOlmoMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            float8_cfg=float8_cfg,
        )
        self.post_feedforward_layernorm = ConceptOlmoRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        attention_output = self.self_attention(hidden_states, position_embeddings, seq_ctx)["projected_output"]
        hidden_states = hidden_states + self.post_attention_layernorm(attention_output)
        mlp_output = self.mlp(hidden_states)
        return hidden_states + self.post_feedforward_layernorm(mlp_output)
