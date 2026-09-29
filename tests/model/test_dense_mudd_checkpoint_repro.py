"""Qwen3.5 dense MUDD 与 reentrant checkpoint 的真实训练回归测试。

TestDenseMUDDCheckpoint
    test_mudd_route_trains_under_full_recompute: 全层重算下 MUDD 路由可完成一个训练步。
    test_mudd_triton_route_trains_under_full_recompute: Triton 路由路径同样可完成训练步。

MUDD 给解码层传入的 ``mudd_streams`` 是嵌套的 4 张量元组，且当前层张量同时以位置参数
出现。原生 reentrant checkpoint 只打包一层，会把该元组当作单个不透明参数、不 detach
其中的张量，于是反向时梯度顺着它们自己的图回到外层图，触发
``Trying to backward through the graph a second time``。本测试以 ``recompute_ratio=1.0``
打开全层重算来覆盖该路径。
"""

import math
import unittest

import torch
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDenseConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig


class _TinyDenseMUDDConfig(Qwen3_5_VLTextDenseConfig):
    """Four layers (three GatedDeltaNet + one full attention) with MUDD enabled."""

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
    # ``key_dim * 2 + value_dim == 128 == in_proj_qkv.out_features`` so the routed
    # sources take the row-sliced projection path rather than the fallback.
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


def _build_engine(route_kernel: bool) -> TrainEngine:
    config = _TinyDenseMUDDConfig(mudd_route_kernel=route_kernel, lm_loss_cfg=CELossConfig(mode="eager"))
    engine = TrainEngine(
        model_cfg=config,
        optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
        fsdp_cfg=FSDPConfig(
            cpu_offload=False,
            recompute_ratio=1.0,
            torch_compile=False,
        ),
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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDenseMUDDCheckpoint(DeterministicDDPTestCase):
    def test_mudd_route_trains_under_full_recompute(self):
        self.create_pg("cuda")
        engine = _build_engine(route_kernel=False)
        try:
            step_info = engine.train_step([_model_item(engine, 2)])
            assert math.isfinite(step_info["total_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_mudd_triton_route_trains_under_full_recompute(self):
        self.create_pg("cuda")
        engine = _build_engine(route_kernel=True)
        try:
            step_info = engine.train_step([_model_item(engine, 2)])
            assert math.isfinite(step_info["total_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_router_w2_stays_zero_after_model_weight_init(self):
        # 回归:default_init_weights 会对名字不含 "norm" 的子模块 .weight 施加
        # normal_(0, 0.02)。router_w2 必须是零,否则零初始化恒等契约失效。
        self.create_pg("cuda")
        engine = _build_engine(route_kernel=False)
        try:
            for route in engine.model.mudd.routes:
                weight = route.router_w2
                if isinstance(weight, DTensor):
                    weight = weight.full_tensor()
                assert torch.equal(weight, torch.zeros_like(weight))
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 1


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestDenseMUDDShardedProjection(DeterministicDDPTestCase):
    def test_row_sliced_projection_is_used_in_a_real_fsdp_forward(self):
        # 钉住行切片路径:FSDP2 在 ``GatedDeltaNet.forward`` 期间把 ``in_proj_qkv.weight``
        # 完整物化(实测非 DTensor,行数与融合宽度相等),守卫成立、切片路径生效,每个
        # routed source 只算自己那一段。一旦退回整权重投影,MUDD 的投影开销就变成
        # vanilla 的三倍——这里让整权重投影直接抛错来钉住它。
        self.create_pg("cuda")
        engine = _build_engine(route_kernel=False)
        try:
            layer = engine.model.layers["0"]
            layer = getattr(layer, "_checkpoint_wrapped_module", layer)
            attn = layer.self_attn
            original_forward = attn.in_proj_qkv.forward

            def _fail_whole_weight(*args, **kwargs):
                raise AssertionError("MUDD 前向调用了整权重投影,投影开销会变成 vanilla 的三倍")

            attn.in_proj_qkv.forward = _fail_whole_weight
            try:
                step_info = engine.train_step([_model_item(engine, 2)])
            finally:
                attn.in_proj_qkv.forward = original_forward
            assert math.isfinite(step_info["total_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 2
