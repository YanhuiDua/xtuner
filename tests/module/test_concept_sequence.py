"""ConceptLM packed 序列变换（``concept_sequence``）回归测试。

这些 helper 决定 token 与"概念"之间怎么换算，参考实现
（``conceptlm/sequence.py``）的语义必须逐条保住，XTuner 侧额外要容忍**不按 chunk 对齐**
的 packed 文档（dolma3 数据里是 ``[8191, 1]`` 这种形状）：

TestConceptSequenceMerge
    test_document_lengths_reads_packed_boundaries: 从 cu_seq_lens 读回每个文档的 token 数。
    test_meanpooling_first_last: 三种归并方式取值与手算一致。
    test_unaligned_document_pools_aligned_prefix: 未对齐文档只归并 chunk 对齐前缀。
    test_short_document_collapses_to_mean: 短于一个 chunk 的文档退化为均值单概念。
TestConceptSequenceExpand
    test_shift_feature_shifts_by_chunk_minus_one: shift_feature 时按 chunk-1 右移（前导补零）。
    test_no_shift_repeats_in_place: 关闭 shift 时概念原地重复。
    test_unaligned_tail_tokens_get_zero_features: 未对齐文档的尾部 token 概念特征为零。
TestConceptSequenceHlm
    test_shift_concepts_left_targets_next_concept: HLM 目标是同文档的下一个概念。
    test_successor_mask_is_batch_major: successor mask 为 batch-major 且文档末尾无效。
TestConceptSequenceContext
    test_concept_context_positions_and_bounds: 概念 context 的位置按 chunk 缩放、边界正确。
"""

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.concept_sequence import (
    build_concept_context,
    document_lengths,
    expand_concepts,
    merge_chunks,
    shift_concepts_left,
    successor_mask,
)


def _seq_ctx(lengths: list[int], hidden: torch.Tensor) -> SequenceContext:
    total = sum(lengths)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu = torch.tensor(boundaries, dtype=torch.int32)
    return SequenceContext(
        input_ids=None,
        inputs_embeds=hidden,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
        position_ids=torch.arange(total).unsqueeze(0),
        device=hidden.device,
        num_padding=0,
        sequence_parallel_mesh=None,
    )


class TestConceptSequenceMerge:
    def test_document_lengths_reads_packed_boundaries(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        assert document_lengths(_seq_ctx([8, 4], hidden)) == [8, 4]

    def test_meanpooling_first_last(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        concepts, concept_lengths = merge_chunks(hidden, [8, 4], chunk_size=4, method="meanpooling")
        assert concept_lengths == [2, 1]
        assert torch.allclose(concepts[0, :, 0], torch.tensor([1.5, 5.5, 9.5]))

        first, _ = merge_chunks(hidden, [8, 4], chunk_size=4, method="first")
        last, _ = merge_chunks(hidden, [8, 4], chunk_size=4, method="last")
        assert torch.allclose(first[0, :, 0], torch.tensor([0.0, 4.0, 8.0]))
        assert torch.allclose(last[0, :, 0], torch.tensor([3.0, 7.0, 11.0]))

    def test_unaligned_document_pools_aligned_prefix(self):
        hidden = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
        concepts, concept_lengths = merge_chunks(hidden, [9], chunk_size=4)
        assert concept_lengths == [2]
        assert torch.allclose(concepts[0, :, 0], torch.tensor([1.5, 5.5]))

    def test_short_document_collapses_to_mean(self):
        hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        concepts, concept_lengths = merge_chunks(hidden, [3, 1], chunk_size=4)
        assert concept_lengths == [1, 1]
        assert torch.allclose(concepts[0, :, 0], torch.tensor([1.0, 3.0]))


class TestConceptSequenceExpand:
    def test_shift_feature_shifts_by_chunk_minus_one(self):
        concepts = torch.tensor([1.5, 5.5, 9.5], dtype=torch.float32).reshape(1, 3, 1)
        expanded = expand_concepts(concepts, [8, 4], [2, 1], chunk_size=4, shift_feature=True)
        assert expanded.shape == (1, 12, 1)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 5.5, 0.0, 0.0, 0.0, 9.5])
        assert torch.allclose(expanded[0, :, 0], expected)

    def test_no_shift_repeats_in_place(self):
        concepts = torch.tensor([1.5, 5.5, 9.5], dtype=torch.float32).reshape(1, 3, 1)
        expanded = expand_concepts(concepts, [8, 4], [2, 1], chunk_size=4, shift_feature=False)
        expected = torch.tensor([1.5] * 4 + [5.5] * 4 + [9.5] * 4)
        assert torch.allclose(expanded[0, :, 0], expected)

    def test_unaligned_tail_tokens_get_zero_features(self):
        concepts = torch.tensor([1.5, 5.5], dtype=torch.float32).reshape(1, 2, 1)
        expanded = expand_concepts(concepts, [9], [2], chunk_size=4, shift_feature=True)
        assert expanded.shape == (1, 9, 1)
        assert torch.allclose(
            expanded[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 1.5, 5.5, 0.0])
        )


class TestConceptSequenceHlm:
    def test_shift_concepts_left_targets_next_concept(self):
        concepts = torch.tensor([1.5, 5.5, 9.5], dtype=torch.float32).reshape(1, 3, 1)
        target = shift_concepts_left(concepts, [2, 1])
        assert torch.allclose(target[0, :, 0], torch.tensor([5.5, 0.0, 9.5]))

    def test_successor_mask_is_batch_major(self):
        mask = successor_mask([2, 1], batch_size=1, device=torch.device("cpu"))
        assert mask.shape == (1, 3)
        assert mask.tolist() == [[True, False, False]]
        # 只有第一个概念有后继，其余（文档末尾 / 单概念文档）都不参与 HLM
        concepts = torch.tensor([[1.5, 5.5, 9.5]])
        assert concepts[mask].tolist() == [1.5]


class TestConceptSequenceContext:
    def test_concept_context_positions_and_bounds(self):
        concepts = torch.zeros(1, 3, 1)
        ctx = build_concept_context(concepts, [2, 1], chunk_size=4)
        assert ctx.cu_seq_lens_q.tolist() == [0, 2, 3]
        assert ctx.position_ids.tolist() == [[0, 4, 0]]
        assert ctx.inputs_embeds.shape == (1, 3, 1)
        assert int(ctx.max_length_q) == 2
