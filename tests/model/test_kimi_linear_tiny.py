"""Smoke tests for Kimi Linear tiny — load / build / short forward (no parity)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig


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
