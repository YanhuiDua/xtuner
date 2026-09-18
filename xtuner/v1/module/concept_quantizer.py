# Copyright (c) OpenMMLab. All rights reserved.
"""Product-vector quantizer for the ConceptLM concept stream.

Port of ``megatron/core/models/conceptlm/quantizer.py`` from the reference
implementation (``jxxiong/residual`` of ``megatron_ncp_qwen35``) so the XTuner
ConceptLM recipe keeps the same concept quantisation contract:

* the hidden vector is split into ``num_codebooks`` heads of ``hidden / num_codebooks``;
* each head selects its nearest code in its own ``codebook_size`` entry table (plain L2);
* the loss is the standard VQ objective ``mse(q, sg(x)) + commitment * mse(sg(q), x)``
  reduced over the valid concept positions.

The codebooks are kept as a ``ParameterList`` of ``[codebook_size, head_dim]``
matrices (instead of one stacked tensor) so Muon keeps treating each codebook as
an independent matrix, which is what the reference implementation relies on.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class QuantizedConcepts:
    """Result of product quantization."""

    vectors: Tensor
    code_ids: Tensor
    loss: Tensor
    numerator: Tensor
    count: Tensor


class ConceptQuantizer(nn.Module):
    """Per-head product quantizer with split codebooks."""

    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int = 32,
        codebook_size: int = 128,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_codebooks <= 0 or codebook_size <= 0:
            raise ValueError("hidden_size, num_codebooks and codebook_size must be positive")
        if hidden_size % num_codebooks:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_codebooks ({num_codebooks})"
            )
        self.hidden_size = int(hidden_size)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.head_dim = self.hidden_size // self.num_codebooks
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.ParameterList(
            [nn.Parameter(torch.empty(self.codebook_size, self.head_dim)) for _ in range(self.num_codebooks)]
        )
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02, truncated: bool = False) -> None:
        for entry in self.codebook:
            if truncated:
                nn.init.trunc_normal_(entry, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
            else:
                nn.init.normal_(entry, mean=0.0, std=std)

    def transformed_codebook(self) -> Tensor:
        return torch.stack(tuple(self.codebook), dim=0)

    def decode_logits(self, logits: Tensor, mode: str = "raw_logits") -> Tensor:
        """Convert per-codebook logits ``[S, B, H, K]`` into concept vectors."""

        if logits.dim() != 4:
            raise ValueError(
                "logits must have shape [S, B, num_codebooks, codebook_size], "
                f"got {tuple(logits.shape)}"
            )
        if logits.shape[2] != self.num_codebooks or logits.shape[3] != self.codebook_size:
            raise ValueError("logits codebook dimensions do not match the quantizer")
        codebook = self.transformed_codebook().to(device=logits.device, dtype=logits.dtype)
        mode = str(mode).lower()
        if mode == "raw_logits":
            weights = logits
        elif mode == "softmax":
            weights = logits.softmax(dim=-1)
        elif mode == "hard_top1":
            ids = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(-1, ids, 1.0)
        else:
            raise ValueError(f"unsupported concept merge mode: {mode!r}")
        vectors = torch.einsum("sbhn,hnd->sbhd", weights, codebook)
        return vectors.reshape(logits.shape[0], logits.shape[1], self.hidden_size)

    def forward(self, concept_hidden: Tensor, concept_mask: Tensor | None = None) -> QuantizedConcepts:
        if concept_hidden.dim() != 3 or concept_hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"concept_hidden must have shape [S, B, {self.hidden_size}], got {tuple(concept_hidden.shape)}"
            )
        concept_seq, batch_size, _ = concept_hidden.shape
        inputs = concept_hidden.reshape(concept_seq, batch_size, self.num_codebooks, self.head_dim)
        codebook = self.transformed_codebook()
        distances = (
            inputs.float().square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.einsum("sbhd,hkd->sbhk", inputs.float(), codebook.float())
            + codebook.float().square().sum(dim=-1).view(1, 1, self.num_codebooks, self.codebook_size)
        )
        code_ids = torch.argmin(distances, dim=-1)
        heads = [
            codebook[idx]
            .index_select(0, code_ids[..., idx].reshape(-1))
            .reshape(concept_seq, batch_size, self.head_dim)
            for idx in range(self.num_codebooks)
        ]
        quantized = torch.stack(heads, dim=2).reshape_as(concept_hidden).to(concept_hidden.dtype)
        per_tensor = F.mse_loss(
            quantized.float(), concept_hidden.detach().float(), reduction="none"
        ).mean(dim=-1)
        per_tensor = per_tensor + self.commitment_cost * F.mse_loss(
            quantized.detach().float(), concept_hidden.float(), reduction="none"
        ).mean(dim=-1)
        if concept_mask is None:
            numerator = per_tensor.sum()
            count = concept_hidden.new_tensor(per_tensor.numel(), dtype=torch.float32)
        else:
            if concept_mask.dtype != torch.bool or tuple(concept_mask.shape) != (concept_seq, batch_size):
                raise ValueError("concept_mask must be a bool tensor shaped [concept_seq, batch]")
            numerator = per_tensor.masked_select(concept_mask).sum()
            count = concept_mask.sum().to(dtype=torch.float32)
        loss = numerator / count.clamp_min(1.0)
        return QuantizedConcepts(quantized, code_ids, loss, numerator, count)
