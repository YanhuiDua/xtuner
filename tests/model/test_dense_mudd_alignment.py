"""模型级 MUDD 结构性与对齐回归。

模块级用例（``tests/module/test_mudd.py``）覆盖了初始化、路由公式与梯度；这里补的是只有
真实模型才成立的"结构性"检查——4 层 tiny 模型（3 个 GatedDeltaNet + 1 个 full attention）
走完整的 TrainEngine / FSDP / checkpoint 栈：

TestDenseMUDDStep0Alignment
    test_zero_init_step0_matches_vanilla: 零初始化下 MUDD 开/关的逐层 hidden、logits、loss 对齐。
TestDenseMUDDStructure
    test_route_sources_are_full_history_in_order: 第 k 层的 sources 是 [embedding, raw_0..raw_k]，
        共 k+2 个且顺序正确（``num_sources = layer_idx + 2``）。
    test_mha_layer_receives_mudd_streams: full attention 层确实收到 MUDD 三路输入；关闭时一个都没有。
    test_mudd_parameters_receive_gradients: 训练步后所有 ``mudd.*`` 参数都有有限梯度且被 optimizer 覆盖。
    test_mudd_with_recompute_mtp_and_grad_accumulation: recompute + MTP + 两个 micro-batch 累积仍跑通。
"""

import math
import unittest

import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDenseConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.mtp import MTPConfig


# FP 阈值。tiny 模型上实测 MUDD 开/关在零初始化时逐层 hidden、logits、loss 的差值全为 0.0
# （逐位一致，见用例里的 [step0-align] 打印）；这里仍留一个 bf16 分辨率量级的容差，以免换
# GPU/换 kernel 选择后因为浮点末位差异误报。真实的接线错误会产生 O(1) 量级的偏差，与此相差
# 两三个数量级。
HIDDEN_DIFF_ATOL = 1.0e-2
LOGITS_REL_ATOL = 1.0e-3
LOSS_REL_ATOL = 1.0e-3


class _TinyDenseMUDDConfig(Qwen3_5_VLTextDenseConfig):
    """Four layers (three GatedDeltaNet + one full attention), MUDD switchable."""

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
    # key_dim * 2 + value_dim == in_proj_qkv.out_features，让 routed source 走行切片投影。
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=4,
        num_key_heads=2,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1.0e-6,
    )
    mudd_enabled: bool = True
    mudd_router_norm_affine: bool = False
    mudd_route_kernel: bool = False


def _build_engine(
    *,
    route_kernel: bool = False,
    mudd_enabled: bool = True,
    recompute_ratio: float = 1.0,
    return_hidden_states: bool = False,
    mtp: bool = False,
) -> TrainEngine:
    config = _TinyDenseMUDDConfig(
        mudd_route_kernel=route_kernel,
        mudd_enabled=mudd_enabled,
        return_hidden_states=return_hidden_states,
        lm_loss_cfg=CELossConfig(mode="eager"),
    )
    if mtp:
        config.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=0.1)
    engine = TrainEngine(
        model_cfg=config,
        optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
        fsdp_cfg=FSDPConfig(cpu_offload=False, recompute_ratio=recompute_ratio, torch_compile=False),
        intra_layer_micro_batch=1,
    )
    engine.init_model_weights()
    return engine


def _model_item(engine: TrainEngine, start: int) -> ModelItem:
    input_ids = torch.arange(start, start + 17).view(1, -1) % engine.model_cfg.vocab_size
    seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],), device="cuda")
    data = {"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]}
    loss_ctx = engine.model.build_loss_ctx_batch([data], sp_mesh=None)[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)


def _copy_base_weights(source: TrainEngine, target: TrainEngine) -> None:
    """Copy every parameter the target owns from the source, so MUDD is the only difference."""
    source_params = dict(source.model.named_parameters())
    copied = 0
    for name, parameter in target.model.named_parameters():
        assert name in source_params, f"target parameter {name} is missing in the source model"
        with torch.no_grad():
            parameter.copy_(source_params[name])
        copied += 1
    assert copied > 0


def _forward(engine: TrainEngine, item: ModelItem):
    with torch.no_grad():
        return engine.model(seq_ctx=item["seq_ctx"], loss_ctx=item["loss_ctx"])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDenseMUDDStep0Alignment(DeterministicDDPTestCase):
    def test_zero_init_step0_matches_vanilla(self):
        # 零初始化时 router_w2 与 static_a 全 0，四路流严格退化为 raw，因此 MUDD 开/关的
        # 逐层 hidden 应该只差"行切片投影 vs 整权重投影"带来的浮点噪声。
        self.create_pg("cuda")
        engine_mudd = _build_engine(mudd_enabled=True, return_hidden_states=True)
        engine_plain = _build_engine(mudd_enabled=False, return_hidden_states=True)
        try:
            _copy_base_weights(engine_mudd, engine_plain)
            item = _model_item(engine_mudd, 2)
            out_mudd = _forward(engine_mudd, item)
            out_plain = _forward(engine_plain, item)

            assert out_mudd.hidden_states is not None and out_plain.hidden_states is not None
            assert len(out_mudd.hidden_states) == len(out_plain.hidden_states) == 4

            per_layer = []
            for layer_idx, (mudd_hidden, plain_hidden) in enumerate(
                zip(out_mudd.hidden_states, out_plain.hidden_states)
            ):
                diff = (mudd_hidden.float() - plain_hidden.float()).abs().max().item()
                per_layer.append(diff)
                assert diff <= HIDDEN_DIFF_ATOL, (
                    f"layer {layer_idx} hidden differs by {diff:.3e} > {HIDDEN_DIFF_ATOL:.1e}; "
                    f"per-layer diffs={per_layer}"
                )

            assert out_mudd.logits is not None and out_plain.logits is not None
            logits_diff = (out_mudd.logits.float() - out_plain.logits.float()).abs().max().item()
            logits_scale = out_plain.logits.float().abs().max().item()
            assert logits_diff <= LOGITS_REL_ATOL * logits_scale, f"logits diff {logits_diff:.3e}"

            assert out_mudd.loss is not None and out_plain.loss is not None
            loss_gap = abs(out_mudd.loss.item() - out_plain.loss.item())
            assert loss_gap <= LOSS_REL_ATOL * abs(out_plain.loss.item()), f"loss diff {loss_gap:.3e}"
            print(
                f"[step0-align] per-layer hidden max_abs={['%.3e' % diff for diff in per_layer]} "
                f"logits_max_abs={logits_diff:.3e} (scale {logits_scale:.3e}) loss_gap={loss_gap:.3e}"
            )
        finally:
            del engine_mudd, engine_plain
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 1


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDenseMUDDStructure(DeterministicDDPTestCase):
    def test_route_sources_are_full_history_in_order(self):
        # 钉住 routes[layer_idx].num_sources == layer_idx + 2，以及 sources 的内容与顺序：
        # [embedding, raw_0, ..., raw_{k-1}] + [raw_k]，off-by-one 会直接破坏历史语义。
        self.create_pg("cuda")
        engine = _build_engine(mudd_enabled=True)
        try:
            mudd = engine.model.mudd
            recorded: list[tuple[int, torch.Tensor, list[torch.Tensor]]] = []
            original_finish_layer = mudd.finish_layer

            def spy_finish_layer(layer_idx, current_raw, state):
                recorded.append(
                    (
                        int(layer_idx),
                        current_raw.detach().clone(),
                        [source.detach().clone() for source in [*state.history, current_raw]],
                    )
                )
                return original_finish_layer(layer_idx, current_raw, state)

            mudd.finish_layer = spy_finish_layer
            try:
                engine.train_step([_model_item(engine, 2)])
            finally:
                mudd.finish_layer = original_finish_layer

            assert [layer_idx for layer_idx, _, _ in recorded] == [0, 1, 2, 3]
            for layer_idx, current_raw, sources in recorded:
                assert len(sources) == layer_idx + 2, f"layer {layer_idx} got {len(sources)} sources"
                assert mudd.routes[layer_idx].num_sources == layer_idx + 2
                assert torch.equal(sources[-1], current_raw), "current layer output must be the last source"
                for prev_idx in range(layer_idx):
                    assert torch.equal(sources[prev_idx + 1], recorded[prev_idx][1]), (
                        f"layer {layer_idx} source {prev_idx + 1} is not raw_{prev_idx}"
                    )
            # sources[0] 是 embedding，与第 0 层的 current_raw 不同。
            assert not torch.equal(recorded[0][2][0], recorded[0][2][1])
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_mha_layer_receives_mudd_streams(self):
        # tiny 模型的第 4 层（idx 3）是 full attention。MUDD 开时必须收到三路输入；
        # 关闭时同样一层必须收到 None。
        self.create_pg("cuda")
        seen_mudd: list[bool] = []
        seen_plain: list[bool] = []

        engine_mudd = _build_engine(mudd_enabled=True)
        engine_plain = _build_engine(mudd_enabled=False)
        try:
            for engine, seen in ((engine_mudd, seen_mudd), (engine_plain, seen_plain)):
                layer = engine.model.layers["3"]
                layer = getattr(layer, "_checkpoint_wrapped_module", layer)
                attention = layer.self_attn
                original = attention.forward

                def spy(*args, __original=original, __seen=seen, **kwargs):
                    __seen.append(kwargs.get("mudd_inputs") is not None)
                    return __original(*args, **kwargs)

                attention.forward = spy
                try:
                    engine.train_step([_model_item(engine, 2)])
                finally:
                    attention.forward = original

            assert seen_mudd and all(seen_mudd), f"full attention layer did not receive MUDD inputs: {seen_mudd}"
            assert seen_plain and not any(seen_plain), f"MUDD disabled but streams were passed: {seen_plain}"
        finally:
            del engine_mudd, engine_plain
            torch.cuda.empty_cache()

    def test_mudd_parameters_receive_gradients(self):
        # 清单第 11 项：MUDD 参数必须全部参与训练，既不能被 FSDP 跳过，也不能漏出 optimizer。
        self.create_pg("cuda")
        engine = _build_engine(mudd_enabled=True)
        try:
            engine.train_step([_model_item(engine, 2)])
            optimizer_param_ids = {
                id(parameter) for group in engine.optimizer.param_groups for parameter in group["params"]
            }
            checked = 0
            for name, parameter in engine.model.named_parameters():
                if "mudd" not in name:
                    continue
                checked += 1
                assert parameter.grad is not None, f"{name} has no gradient"
                assert torch.isfinite(parameter.grad.float()).all(), f"{name} gradient is not finite"
                assert id(parameter) in optimizer_param_ids, f"{name} is not covered by the optimizer"
            assert checked >= 4 * 4, f"expected at least one route per layer, saw {checked} MUDD parameters"
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_mudd_with_recompute_mtp_and_grad_accumulation(self):
        # 清单第 14 项：重算 + MTP + 梯度累积同时打开时 MUDD 仍要正确完成训练步。
        self.create_pg("cuda")
        engine = _build_engine(mudd_enabled=True, recompute_ratio=1.0, mtp=True)
        try:
            assert engine.grad_accumulation_steps(2) == 2
            step_info = engine.train_step([_model_item(engine, 2), _model_item(engine, 11)])
            assert math.isfinite(step_info["total_loss"])
            for name, parameter in engine.model.named_parameters():
                if "mudd" not in name:
                    continue
                assert parameter.grad is None or torch.isfinite(parameter.grad.float()).all()
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 1
