"""ConceptLM 残差 VQ 量化器（``ConceptQuantizer``）回归测试。

量化器是 ConceptLM 配方里唯一不做前后向近似的新模块，参考实现
（``megatron/core/models/conceptlm/quantizer.py``）的语义必须逐条保住：

TestConceptQuantizerForward
    test_code_ids_match_brute_force: 逐头最近邻码字与暴力解一致。
    test_vq_loss_matches_manual_formula: VQ loss = mse(q, sg(x)) + commitment * mse(sg(q), x)。
    test_masked_reduction_only_counts_valid_concepts: 掩码归约只统计有效概念位置。
    test_gradients_reach_input_and_codebooks: 梯度可达输入与码本。
TestConceptQuantizerDecode
    test_decode_modes_shapes: raw_logits / softmax / hard_top1 三种融合模式形状正确。
    test_hard_top1_selects_argmax_code: hard_top1 恰好取 argmax 码字。
TestConceptQuantizerContract
    test_two_b_recipe_shape: 32 个码本 × 128 码字 / hidden 2048 的真实配方形状可跑通。
    test_init_extra_weights_covers_codebooks: ``init_extra_weights`` 覆盖全部码本参数名
        （XTuner 要求每个参数都被初始化器或该钩子覆盖）。
"""

import pytest
import torch

from xtuner.v1.module.concept_quantizer import ConceptQuantizer


def _brute_force_code_ids(x: torch.Tensor, quantizer: ConceptQuantizer) -> torch.Tensor:
    """逐头显式计算最近邻码字 id。"""
    ids = []
    for head in range(quantizer.num_codebooks):
        part = x[..., head * quantizer.head_dim : (head + 1) * quantizer.head_dim]
        dist = ((part.unsqueeze(-2) - quantizer.codebook[head]) ** 2).sum(-1)
        ids.append(dist.argmin(-1))
    return torch.stack(ids, dim=-1)


class TestConceptQuantizerForward:
    def test_code_ids_match_brute_force(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4)
        x = torch.randn(3, 1, 8)
        out = quantizer(x)
        assert out.code_ids.shape == (3, 1, 2)
        assert torch.equal(out.code_ids, _brute_force_code_ids(x, quantizer))
        assert out.vectors.shape == x.shape

    def test_vq_loss_matches_manual_formula(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4, commitment_cost=0.25)
        x = torch.randn(3, 1, 8)
        out = quantizer(x)
        per_position = ((out.vectors.detach() - x) ** 2).mean(-1)
        expected = per_position + 0.25 * per_position
        assert torch.allclose(out.loss, expected.mean(), atol=1e-6)
        assert float(out.count) == float(per_position.numel())

    def test_masked_reduction_only_counts_valid_concepts(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4, commitment_cost=0.25)
        x = torch.randn(3, 1, 8)
        out = quantizer(x)
        per_position = ((out.vectors.detach() - x) ** 2).mean(-1)
        per_position = per_position + 0.25 * per_position
        mask = torch.tensor([[True], [False], [True]])
        masked = quantizer(x, mask)
        assert torch.allclose(masked.loss, per_position.detach()[mask].mean(), atol=1e-6)
        assert float(masked.count) == 2.0
        assert masked.numerator.shape == ()

    def test_gradients_reach_input_and_codebooks(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4)
        x = torch.randn(3, 1, 8, requires_grad=True)
        quantizer(x).loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        grads = [p.grad for p in quantizer.codebook if p.grad is not None]
        assert len(grads) == quantizer.num_codebooks
        assert all(torch.isfinite(g).all() for g in grads)


class TestConceptQuantizerDecode:
    @pytest.mark.parametrize("mode", ["raw_logits", "softmax", "hard_top1"])
    def test_decode_modes_shapes(self, mode: str):
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4)
        logits = torch.randn(3, 1, 2, 4)
        vectors = quantizer.decode_logits(logits, mode=mode)
        assert vectors.shape == (3, 1, 8)

    def test_hard_top1_selects_argmax_code(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=8, num_codebooks=2, codebook_size=4)
        logits = torch.randn(3, 1, 2, 4)
        decoded = quantizer.decode_logits(logits, mode="hard_top1")
        picked = logits.argmax(-1)
        expected = torch.stack([quantizer.codebook[h][picked[..., h]] for h in range(2)], dim=-2)
        assert torch.allclose(decoded, expected.reshape(3, 1, 8))


class TestConceptQuantizerContract:
    def test_two_b_recipe_shape(self):
        torch.manual_seed(0)
        quantizer = ConceptQuantizer(hidden_size=2048, num_codebooks=32, codebook_size=128)
        x = torch.randn(4, 1, 2048)
        out = quantizer(x)
        assert out.vectors.shape == x.shape
        assert int(out.code_ids.min()) >= 0
        assert int(out.code_ids.max()) < 128
        assert torch.isfinite(out.loss)
        assert quantizer.transformed_codebook().shape == (32, 128, 64)

    def test_init_extra_weights_covers_codebooks(self):
        quantizer = ConceptQuantizer(hidden_size=16, num_codebooks=4, codebook_size=8)
        for name, param in quantizer.named_parameters():
            torch.nn.init.zeros_(param)
        initialized = quantizer.init_extra_weights()
        assert initialized == {f"codebook.{idx}" for idx in range(4)}
        for codebook in quantizer.codebook:
            assert torch.isfinite(codebook).all()
            assert codebook.abs().sum() > 0
