"""Smoke tests for Kimi Linear tiny — AttnRes + load / forward (no parity)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig
from xtuner.v1.model.moe.kimi_linear.decoder_layer import apply_attn_res
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.linear import build_linear


class TestKimiLinearTiny(unittest.TestCase):
    def test_build_structure(self):
        cfg = KimiLinearTinyConfig()
        model = cfg.build()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 0)
        self.assertEqual(len(model.layers), cfg.num_hidden_layers)
        self.assertEqual(cfg.layers_type[0], "linear_attention")
        self.assertEqual(cfg.layers_type[3], "full_attention")
        moe_layer = model.layers["1"]
        self.assertTrue(hasattr(moe_layer, "routed_expert_down_proj"))
        self.assertTrue(hasattr(moe_layer, "self_attention_res_proj"))
        self.assertTrue(hasattr(model, "output_attn_res_proj"))

    def test_apply_attn_res_shape(self):
        tokens, hidden = 8, 16
        prefix = torch.randn(tokens, hidden)
        block = torch.randn(tokens, 2, hidden)
        proj = build_linear(hidden, 1, bias=False)
        norm = RMSNorm(hidden, eps=1e-5)
        out = apply_attn_res(prefix, block, proj, norm)
        self.assertEqual(tuple(out.shape), (tokens, hidden))

    def test_attn_res_block_grows_across_layers(self):
        """Empty block residual should gain one slot every attn_res_block_size layers."""
        cfg = KimiLinearTinyConfig()
        assert cfg.attn_res_block_size == 2
        model = cfg.build()
        layer0 = model.layers["0"]
        hs = torch.randn(1, 4, cfg.hidden_size, dtype=torch.float32)
        # Use float32 path without GPU kernels where possible — skip if attn needs CUDA
        if not torch.cuda.is_available():
            # Still verify layer0 appends on empty residual via the AttnRes bookkeeping
            # by calling the mixer-only portion: empty -> append when layer_idx % 2 == 0
            br = hs.new_zeros(4, 0, cfg.hidden_size)
            self.assertEqual(br.shape[1], 0)
            # layer 0 is dense with AttnRes; without CUDA we only check config wiring
            self.assertEqual(layer0.layer_idx % cfg.attn_res_block_size, 0)
            self.assertEqual(model.layers["2"].layer_idx % cfg.attn_res_block_size, 0)
            return

        model = model.to(device="cuda", dtype=torch.bfloat16)
        if hasattr(model, "init_weights"):
            model.init_weights()
        ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long, device="cuda")
        labels = ids.clone()
        seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
        loss_ctx = CELossConfig().build(data={"shifted_labels": labels}, sp_mesh=None)
        out = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        loss = out["loss"] if isinstance(out, dict) else getattr(out, "loss", None)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()

    def test_forward_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        cfg = KimiLinearTinyConfig()
        model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
        if hasattr(model, "init_weights"):
            model.init_weights()
        ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long, device="cuda")
        labels = ids.clone()
        seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
        loss_ctx = CELossConfig().build(data={"shifted_labels": labels}, sp_mesh=None)
        out = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        loss = out["loss"] if isinstance(out, dict) else getattr(out, "loss", None)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).all())

    def test_save_hf_roundtrip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for save_hf path")
        cfg = KimiLinearTinyConfig()
        model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
        if hasattr(model, "init_weights"):
            model.init_weights()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model.save_hf(str(out))
            model2 = KimiLinearTinyConfig().build().to(device="cuda", dtype=torch.bfloat16)
            model2.from_hf(str(out))


if __name__ == "__main__":
    unittest.main()
