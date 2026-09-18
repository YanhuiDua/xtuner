# Copyright (c) OpenMMLab. All rights reserved.
"""Packed-sequence helpers for the ConceptLM concept stream.

XTuner keeps a packed batch as ``[1, total_tokens, hidden]`` plus
``cu_seq_lens_q`` document boundaries, so the token<->concept transforms have to
be document aware.  These helpers port the behaviour of the reference
implementation (``conceptlm/sequence.py``):

* ``merge_chunks`` reduces every ``chunk_size`` consecutive tokens of a document
  to one concept state (``meanpooling`` / ``first`` / ``last``); like the
  reference implementation every packed document must be chunk aligned;
* ``expand_concepts`` repeats concepts back to token resolution and, with
  ``shift_feature``, shifts the block right by ``chunk_size - 1`` so a token only
  sees the *previous* chunk's concept (reference ``shift_sequences(right)``);
* ``shift_concepts_left`` / ``successor_mask`` define the HLM target: the target
  of concept ``c`` is concept ``c + 1`` of the same document.
"""

from itertools import accumulate
from typing import Literal, cast
import torch
from torch import Tensor
from xtuner.v1.data_proto import SequenceContext


def document_lengths(seq_ctx: SequenceContext) -> list[int]:
    """Per-document token counts of a packed batch."""
    if seq_ctx.cu_seq_lens_q.numel() == 2 and seq_ctx.num_padding == 0:
        return [int(seq_ctx.max_length_q.item())]
    lengths = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
    return [int(length) for length in lengths.detach().cpu().tolist()]


def merge_chunks(
    hidden_states: Tensor,
    doc_lengths: list[int],
    chunk_size: int,
    method: Literal["meanpooling", "first", "last"] = "meanpooling",
) -> tuple[Tensor, list[int]]:
    """Reduce token-resolution states to one state per ``chunk_size`` tokens."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if any(length % chunk_size for length in doc_lengths):
        raise ValueError(
            f"every packed document must align to chunk_size={chunk_size}, got lengths {doc_lengths}"
        )
    chunks: list[Tensor] = []
    concept_lengths: list[int] = []
    start = 0
    for seq_len in doc_lengths:
        state = hidden_states[:, start : start + seq_len]
        start += seq_len
        if seq_len == 0:
            raise ValueError("empty packed documents are not supported")
        reshaped = state.reshape(1, seq_len // chunk_size, chunk_size, *state.shape[2:])
        if method == "first":
            merged = reshaped[:, :, 0]
        elif method == "last":
            merged = reshaped[:, :, -1]
        else:
            merged = reshaped.mean(dim=2)
        chunks.append(merged)
        concept_lengths.append(merged.shape[1])
    if len(chunks) == 1:
        return chunks[0].contiguous(), concept_lengths
    return torch.cat(chunks, dim=1), concept_lengths


def shift_concepts_left(concept_states: Tensor, concept_lengths: list[int]) -> Tensor:
    """Target for the HLM objective: concept ``c + 1`` inside each document."""
    shifted: list[Tensor] = []
    start = 0
    for length in concept_lengths:
        state = concept_states[:, start : start + length]
        start += length
        if length == 1:
            document = state
        else:
            document = torch.cat((state[:, 1:], torch.zeros_like(state[:, :1])), dim=1)
        shifted.append(document)
    if len(shifted) == 1:
        return shifted[0].contiguous()
    return torch.cat(shifted, dim=1)


def successor_mask(concept_lengths: list[int], batch_size: int, device: torch.device) -> Tensor:
    """``True`` where a concept has a valid successor inside its document."""
    mask: list[Tensor] = []
    for length in concept_lengths:
        valid = torch.zeros(length, batch_size, dtype=torch.bool, device=device)
        if length > 1:
            valid[:-1] = True
        mask.append(valid)
    if len(mask) == 1:
        return mask[0]
    return torch.cat(mask, dim=0)


def expand_concepts(
    concept_states: Tensor,
    doc_lengths: list[int],
    concept_lengths: list[int],
    chunk_size: int,
    shift_feature: bool = True,
) -> Tensor:
    """Repeat concept states back to token resolution (see module docstring)."""
    repeated_docs: list[Tensor] = []
    start = 0
    for concept_len, seq_len in zip(concept_lengths, doc_lengths, strict=True):
        state = concept_states[:, start : start + concept_len]
        start += concept_len
        repeated = state.repeat_interleave(chunk_size, dim=1)
        if repeated.shape[1] != seq_len:
            raise ValueError(
                f"concept expansion produced {repeated.shape[1]} tokens for a document of {seq_len}"
            )
        if shift_feature and chunk_size > 1:
            shift = chunk_size - 1
            repeated = torch.cat((torch.zeros_like(repeated[:, :shift]), repeated[:, :-shift]), dim=1)
        repeated_docs.append(repeated)
    if len(repeated_docs) == 1:
        return repeated_docs[0].contiguous()
    return torch.cat(repeated_docs, dim=1)


def build_concept_context(
    concept_hidden: Tensor,
    concept_lengths: list[int],
    chunk_size: int,
) -> SequenceContext:
    """SequenceContext for a concept-resolution stack (positions match token positions)."""
    device = concept_hidden.device
    cu_seq_lens = torch.tensor([0, *accumulate(concept_lengths)], dtype=torch.int32).to(device)
    position_ids = torch.cat(
        [torch.arange(length, device=device) * chunk_size for length in concept_lengths], dim=0
    ).unsqueeze(0)
    return SequenceContext(
        input_ids=None,
        inputs_embeds=cast(torch.FloatTensor, concept_hidden),
        cu_seq_lens_q=cast(torch.IntTensor, cu_seq_lens),
        cu_seq_lens_k=cast(torch.IntTensor, cu_seq_lens),
        max_length_q=max(concept_lengths),
        max_length_k=max(concept_lengths),
        position_ids=cast(torch.LongTensor, position_ids),
        device=device,
        num_padding=0,
        sequence_parallel_mesh=None,
    )
