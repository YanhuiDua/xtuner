"""Qwen3.5 ConceptLM 的 MUDD 恒等契约与 packed 兼容性回归。

MUDD 路由零初始化时整段模型必须退化成"没有 MUDD"，参考实现与本仓库 9B dense 都依赖这条
契约。这里在 tiny 的 ConceptLM（2 层 encoder / 1 层 predictor / 2 层 decoder，chunk 4，
4 个码本）上走一次真实前向，比较 MUDD 全关与三段全开的中间张量：

TestConceptLMQwen35Identity
    test_stack_outputs_are_bitwise_identical: encoder / concept / predictor 各段输出逐位一致。
    test_logits_stay_within_bf16_noise: logits 差异停留在 bf16 舍入量级（预测头解码处的 1 ulp
        会经深 decoder 放大，因此只做量级断言）。
    test_packed_unaligned_documents_run: packed 文档不按 chunk 对齐（``[7, 1]``）时前向仍可跑通
        且输出有限。
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.dense.conceptlm_qwen35 import ConceptLMQwen35Config
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig


class _TinyConceptLMConfig(ConceptLMQwen35Config):
    """Small ConceptLM used by the tests (2/1/2 layers, chunk 4, 4 codebooks)."""

    vocab_size: int = 64
    max_position_embeddings: int = 64
    pad_token_id: int | None = 0
    eos_token_id: int = 1
    num_hidden_layers: int = 4
    hidden_size: int = 64
    intermediate_size: int = 128
    rms_norm_eps: float = 1.0e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    attention: MHAConfig = MHAConfig(
        with_gate=True,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        qk_norm=True,
        rms_norm_eps=1.0e-6,
        rms_norm_type="zero_centered",
    )
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=4,
        num_key_heads=2,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1.0e-6,
    )
    concept_encoder_layers: int = 2
    concept_predictor_layers: int = 1
    concept_decoder_layers: int = 2
    concept_chunk_size: int = 4
    concept_codebook_size: int = 8
    concept_num_codebooks: int = 4
    mudd_route_kernel: bool = False
    mudd_router_norm_affine: bool = False


LOGITS_REL_ATOL = 5.0e-2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ConceptLM forward requires CUDA kernels")
class TestConceptLMQwen35Identity:
    @staticmethod
    def _build(mudd: bool, seed: int = 0):
        config = _TinyConceptLMConfig(
            concept_encoder_mudd=mudd,
            concept_predictor_mudd=mudd,
            concept_decoder_mudd=mudd,
            compile_cfg=False,
        )
        torch.manual_seed(seed)
        model = config.build().cuda().to(torch.bfloat16)
        model.eval()
        return model

    @staticmethod
    def _seq_ctx(lengths: list[int], device: torch.device) -> SequenceContext:
        total = sum(lengths)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu = torch.tensor(boundaries, dtype=torch.int32, device=device)
        input_ids = torch.randint(0, 64, (1, total), device=device)
        return SequenceContext(
            input_ids=input_ids,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
            num_padding=0,
            sequence_parallel_mesh=None,
            device=device,
            position_ids=torch.arange(total, device=device).unsqueeze(0),
        )

    def test_stack_outputs_are_bitwise_identical(self):
        model_off = self._build(mudd=False)
        model_on = self._build(mudd=True)
        seq_ctx = self._seq_ctx([8], torch.device("cuda"))

        with torch.no_grad():
            _, off = model_off(seq_ctx, return_intermediates=True)
            _, on = model_on(seq_ctx, return_intermediates=True)

        # 零初始化路由 → 三段栈的输出必须逐位一致（不是"接近"）。
        # ``predicted`` 已经过 32 个头/解码 einsum，MUDD 路径交回的张量 stride 不同会
        # 触发不同的 GEMM kernel，只保证在 bf16 噪声量级内（见下面的断言）。
        for key in ("encoder_hidden", "concept_hidden"):
            assert torch.equal(off[key], on[key]), f"{key} is not bitwise identical"
        layer_keys = [k for k in off if k.startswith("predictor_layer")]
        assert layer_keys, "predictor 逐层记录缺失"
        for key in layer_keys:
            assert torch.equal(off[key], on[key]), f"{key} is not bitwise identical"

        predicted_gap = (off["predicted"].float() - on["predicted"].float()).abs().max()
        predicted_scale = off["predicted"].float().abs().mean().clamp_min(1e-6)
        assert predicted_gap / predicted_scale < LOGITS_REL_ATOL, (
            f"predicted 相对差 {predicted_gap / predicted_scale:.3e} 超出 bf16 噪声量级"
        )

        off_logits = off["decoder_hidden"].float()
        on_logits = on["decoder_hidden"].float()
        rel = (off_logits - on_logits).abs().max() / off_logits.abs().mean().clamp_min(1e-6)
        assert rel < LOGITS_REL_ATOL, f"decoder hidden 相对差 {rel:.3e} 超出 bf16 噪声量级"

    def test_logits_stay_within_bf16_noise(self):
        model_off = self._build(mudd=False)
        model_on = self._build(mudd=True)
        seq_ctx = self._seq_ctx([8], torch.device("cuda"))
        with torch.no_grad():
            logits_off = model_off(seq_ctx)["logits"].float()
            logits_on = model_on(seq_ctx)["logits"].float()
        rel = (logits_off - logits_on).abs().max() / logits_off.abs().mean().clamp_min(1e-6)
        assert rel < LOGITS_REL_ATOL, f"logits 相对差 {rel:.3e} 超出 bf16 噪声量级"

    @pytest.mark.parametrize("mudd", [False, True])
    def test_packed_unaligned_documents_run(self, mudd: bool):
        model = self._build(mudd=mudd)
        # dolma3 里真实出现的形状：8 个 token 切成 7 + 1，第二段短于一个 chunk。
        seq_ctx = self._seq_ctx([7, 1], torch.device("cuda"))
        with torch.no_grad():
            logits = model(seq_ctx)["logits"]
        assert logits.shape == (1, 8, 64)
        assert torch.isfinite(logits.float()).all()
