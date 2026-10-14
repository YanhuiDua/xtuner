"""Kimi Linear tiny tests aligned with ``run_kimi_linear_tiny_8gpu.sh``.

CPU tests cover structure. CUDA tests follow the smoke path: meta build,
FSDP, ``from_hf`` of the gen ckpt (must succeed), next-token CE, AdamW steps.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossConfig, CELossContext
from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig
from xtuner.v1.model.moe.kimi_linear.decoder_layer import apply_attn_res
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.linear import build_linear


def _tiny_ckpt() -> Path:
    return Path(__file__).resolve().parents[2] / "work_dirs" / "kimi_linear_tiny_bf16"


def _require_tiny_ckpt() -> Path:
    ckpt = _tiny_ckpt()
    if not (ckpt / "model.safetensors.index.json").exists() and not any(ckpt.glob("*.safetensors")):
        raise unittest.SkipTest(
            f"missing tiny ckpt at {ckpt}; run "
            ".dev_scripts/gen_kimi_linear_tiny_bf16.py or examples/v1/scripts/run_kimi_linear_tiny_8gpu.sh"
        )
    return ckpt


def _next_token_batch(
    vocab_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (1, seq_len + 1), device=device, dtype=torch.long)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def _lm_loss_ctx(shifted_labels: torch.Tensor) -> CELossContext:
    loss_ctx = CELossConfig().build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    assert loss_ctx is not None
    return CELossContext.build_batches([loss_ctx])[0]


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

    def test_attn_res_layer_schedule(self):
        cfg = KimiLinearTinyConfig()
        self.assertEqual(cfg.attn_res_block_size, 2)
        model = cfg.build()
        self.assertEqual(model.layers["0"].layer_idx % cfg.attn_res_block_size, 0)
        self.assertEqual(model.layers["2"].layer_idx % cfg.attn_res_block_size, 0)


class TestKimiLinearTinyTrain(DeterministicDDPTestCase):
    """8-GPU FSDP next-token train, same path as ``run_kimi_linear_tiny_8gpu.sh``."""

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_fsdp_next_token_train(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if torch.cuda.device_count() < self.world_size:
            self.skipTest(f"need {self.world_size} GPUs, have {torch.cuda.device_count()}")

        self.create_pg("cuda")
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
        cfg = KimiLinearTinyConfig()
        ckpt = _require_tiny_ckpt()
        seq_len = 16
        steps = 2

        with torch.device("meta"):
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        fsdp_cfg = FSDPConfig(ep_size=1, reshard_after_forward=True, cpu_offload=False)
        model.fully_shard(fsdp_config=fsdp_cfg)
        model.from_hf(str(ckpt))

        torch.manual_seed(0 + dist.get_rank())
        input_ids, shifted_labels = _next_token_batch(cfg.vocab_size, seq_len, device)
        params = [p for _, p in model.trainable_parameters()]
        optim = torch.optim.AdamW(params, lr=1e-3, foreach=False)
        model.train()

        losses: list[float] = []
        for step in range(steps):
            seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
            loss_ctx = _lm_loss_ctx(shifted_labels)
            out = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            loss = out["loss"] if isinstance(out, dict) else getattr(out, "loss", None)
            self.assertIsNotNone(loss)
            self.assertTrue(torch.isfinite(loss).all(), f"non-finite loss at step {step}: {loss}")

            loss.backward()
            model.scale_and_reduce_grad()
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().float()))

        dist.barrier()
        self.assertEqual(len(losses), steps)

    def test_save_hf_roundtrip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if torch.cuda.device_count() < self.world_size:
            self.skipTest(f"need {self.world_size} GPUs, have {torch.cuda.device_count()}")

        self.create_pg("cuda")
        cfg = KimiLinearTinyConfig()
        src = _require_tiny_ckpt()
        with torch.device("meta"):
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        fsdp_cfg = FSDPConfig(ep_size=1, reshard_after_forward=True, cpu_offload=False)
        model.fully_shard(fsdp_config=fsdp_cfg)
        model.from_hf(str(src))

        with tempfile.TemporaryDirectory() as tmp:
            syncdir = [tmp]
            dist.broadcast_object_list(syncdir, src=0)
            out_ckpt = Path(syncdir[0])
            model.save_hf(str(out_ckpt))
            dist.barrier()

            with torch.device("meta"):
                model2 = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
            model2.fully_shard(fsdp_config=fsdp_cfg)
            model2.from_hf(str(out_ckpt))
            dist.barrier()


if __name__ == "__main__":
    unittest.main()
