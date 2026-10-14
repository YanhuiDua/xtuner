"""Kimi-K3 tiny HF numerical parity tests.

Compares XTuner ``KimiLinearTiny`` against HuggingFace ``KimiLinearForCausalLM``
(trust_remote_code from ``moonshotai/Kimi-K3``) on a synthetic bf16 tiny checkpoint.

Hub HF MoE raises in training (``KimiSparseMoeBlock`` / ``KimiMoEGate``), so
full-model and MoE-layer checks are eval/forward. The dense first layer also
checks backward. Reference: ``tests/model/test_qwen3_moe.py`` and add_hf_model skill §7.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoModelForCausalLM
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig


_HF_REMOTE_CODE_FILES = (
    "configuration_kimi_k3.py",
    "modeling_kimi_linear.py",
)
_PARITY_TEXTS = [
    "数据应该像山间的清泉，自然地流向它该去的地方",
    "当异常来临时，就像秋风中飘落的叶子， 应该被温柔地接住，而不是粗暴地丢弃",
    "当函数被调用时，它应该像春天的第一缕阳光，温柔地唤醒沉睡的数据结构",
    "就像老树拥抱归巢的鸟儿，内存管理应该给予每个对象足够的安全感",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tiny_ckpt_candidates() -> list[Path]:
    env = os.environ.get("KIMI_LINEAR_TINY_PATH")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.append(_repo_root() / "work_dirs" / "kimi_linear_tiny_bf16")
    return paths


def _remote_code_candidates() -> list[Path]:
    env = os.environ.get("KIMI_K3_HF_CODE")
    roots: list[Path] = []
    if env:
        roots.append(Path(env))
    roots.append(_repo_root() / ".cache" / "kimi_k3_hf")
    roots.extend(_tiny_ckpt_candidates())
    return roots


def _copy_hf_remote_code(dst: Path) -> None:
    """Copy Kimi-K3 trust_remote_code files so HF AutoModel can load the tiny dir."""
    dst.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for name in _HF_REMOTE_CODE_FILES:
        target = dst / name
        if target.exists():
            continue
        src = next((root / name for root in _remote_code_candidates() if (root / name).exists()), None)
        if src is None:
            missing.append(name)
            continue
        shutil.copy2(src, target)
    if missing:
        raise unittest.SkipTest(
            "need Kimi-K3 remote-code files for HF parity: "
            + ", ".join(missing)
            + ". Set KIMI_K3_HF_CODE to a dir containing configuration_kimi_k3.py "
            "and modeling_kimi_linear.py (from moonshotai/Kimi-K3)."
        )


def _bind_official_hf_config_class(config: KimiLinearTinyConfig) -> None:
    source = next(
        (
            root / "configuration_kimi_k3.py"
            for root in _remote_code_candidates()
            if (root / "configuration_kimi_k3.py").is_file()
        ),
        None,
    )
    if source is None:
        raise unittest.SkipTest("need configuration_kimi_k3.py to create a standalone KimiLinear checkpoint")
    spec = importlib.util.spec_from_file_location("_xtuner_test_kimi_k3_config", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official Kimi config implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config._hf_config_cls = module.KimiLinearConfig


def _existing_tiny_ckpt() -> Path | None:
    for path in _tiny_ckpt_candidates():
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path, encoding="utf-8") as file:
                weight_files = set(json.load(file).get("weight_map", {}).values())
            if weight_files and all(
                (path / name).is_file() and os.access(path / name, os.R_OK) for name in weight_files
            ):
                return path
        elif any(weight.is_file() and os.access(weight, os.R_OK) for weight in path.glob("*.safetensors")):
            return path
    return None


def _text_to_ids(text: str, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Map CJK text to the tiny vocab without requiring the Hub tokenizer."""
    ids = [(ord(ch) % (vocab_size - 4)) + 1 for ch in text[:48]]
    if len(ids) < 8:
        ids = ids + list(range(1, 9 - len(ids)))
    return torch.tensor([ids], device=device, dtype=torch.long)


def _xtuner_ce_loss(model, input_ids: torch.Tensor, loss_mode: str = "eager"):
    shift_input_ids = input_ids[:, :-1]
    shifted_labels = input_ids[:, 1:]
    seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))
    loss_cfg = CELossConfig(mode=loss_mode)
    loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    with torch.no_grad():
        output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    return output["loss"]


def _patch_hf_transformers_compat() -> None:
    """Hub modeling targets a slightly older transformers API than 5.14.

    * ``OutputRecorder`` moved from ``utils.generic`` to ``utils.output_capturing``.
    * Some builds removed the deprecated ``check_model_inputs`` alias while
      retaining its replacement, ``merge_with_config_defaults``.
    * ``create_causal_mask`` changed ``input_embeds``/``inputs_embeds`` and
      ``cache_position`` across releases, so adapt to the installed signature.
    """
    import transformers.masking_utils as masking_utils
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder
    if not hasattr(generic, "check_model_inputs"):
        generic.check_model_inputs = generic.merge_with_config_defaults

    orig = masking_utils.create_causal_mask
    if not getattr(orig, "_kimi_k3_compat", False):
        parameters = inspect.signature(orig).parameters

        def create_causal_mask(*args, **kwargs):
            if "inputs_embeds" in parameters and "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            elif "input_embeds" in parameters and "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            if "cache_position" not in parameters:
                kwargs.pop("cache_position", None)
            return orig(*args, **kwargs)

        create_causal_mask._kimi_k3_compat = True  # type: ignore[attr-defined]
        masking_utils.create_causal_mask = create_causal_mask


def _load_hf_causal_lm(ckpt: Path):
    # Hub constructor rewrites attn to flash_attention_2; restore eager so MLA
    # softmax can match XTUNER_HF_IMPL. KDA ignores this flag.
    _patch_hf_transformers_compat()
    hf = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    hf.eval()
    hf.config._attn_implementation = "eager"
    return hf


class TestKimiLinearTinyHFParity(DeterministicDDPTestCase):
    """Single-rank HF vs XTuner accuracy on the tiny Kimi-K3 text tower."""

    def prepare(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))

    def _ckpt_dir(self, require_hf_code: bool = False) -> Path:
        existing = _existing_tiny_ckpt()
        if existing is not None and not require_hf_code:
            return existing

        tmp = Path(self.temp_dir.name) / ("kimi_linear_tiny_hf" if require_hf_code else "kimi_linear_tiny_bf16")
        if dist.is_initialized() and dist.get_world_size() > 1:
            sync = [str(tmp)]
            dist.broadcast_object_list(sync, src=0)
            tmp = Path(sync[0])
            if dist.get_rank() == 0:
                self._materialize_ckpt(tmp, existing, require_hf_code)
            dist.barrier()
        else:
            self._materialize_ckpt(tmp, existing, require_hf_code)
        return tmp

    def _materialize_ckpt(self, tmp: Path, existing: Path | None, require_hf_code: bool) -> None:
        if (tmp / "config.json").exists():
            if require_hf_code:
                _copy_hf_remote_code(tmp)
            return
        tmp.mkdir(parents=True, exist_ok=True)
        if existing is not None:
            for src in existing.iterdir():
                if src.is_file():
                    shutil.copy2(src, tmp / src.name)
        else:
            cfg = KimiLinearTinyConfig()
            _bind_official_hf_config_class(cfg)
            cfg.compile_cfg = False
            model = cfg.build().to(device="cuda", dtype=torch.bfloat16)
            model.init_weights()
            model.save_hf(str(tmp))
            del model
            torch.cuda.empty_cache()
        if require_hf_code:
            _copy_hf_remote_code(tmp)

    def _build_xtuner(self, ckpt: Path, *, dispatcher=None, ep_size: int = 1, fsdp: bool = False):
        with torch.device("meta"):
            # Load through the official remote config class so save_hf can
            # reconstruct a valid standalone kimi_linear config instead of
            # copying legacy tiny configs that lack model_type.
            cfg = KimiLinearTinyConfig.from_hf(ckpt)
            cfg.compile_cfg = False
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        if fsdp:
            model.fully_shard(fsdp_config=FSDPConfig(ep_size=ep_size, cpu_offload=False))
        model.from_hf(str(ckpt))
        model.eval()
        return model

    @parametrize.parametrize(
        "device,dispatcher,ep_size,tol,loss_mode",
        [
            ("cuda", None, 1, 1e-2, "eager"),
            ("cuda", None, 1, 1e-2, "chunk"),
        ],
    )
    def test_kimi_linear_tiny_run(self, device, dispatcher, ep_size, tol, loss_mode):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.create_pg(device)
        ckpt = self._ckpt_dir(require_hf_code=True)
        vocab_size = KimiLinearTinyConfig().vocab_size

        with self.hf_impl():
            hf_model = _load_hf_causal_lm(ckpt)
            expected_losses = []
            input_list = [_text_to_ids(text, vocab_size, torch.device("cuda")) for text in _PARITY_TEXTS]
            for input_ids in input_list:
                with torch.no_grad():
                    output = hf_model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
                expected_losses.append(output.loss)
            del hf_model
            torch.cuda.empty_cache()

            xt_model = self._build_xtuner(ckpt, dispatcher=dispatcher, ep_size=ep_size)
            losses = [_xtuner_ce_loss(xt_model, input_ids, loss_mode=loss_mode) for input_ids in input_list]

        self._check_loss_curve(
            losses=torch.tensor(losses),
            losses_ref=torch.tensor(expected_losses),
            sim_tol=tol,
            rtol=tol,
        )
        dist.barrier()

    @parametrize.parametrize("device,layer_idx", [("cuda", 0), ("cuda", 3)])
    def test_decoder_layer_parity(self, device, layer_idx):
        # Linear=0 (dense KDA) and full=3 (MoE MLA). Under XTUNER_HF_IMPL the
        # layer output must match HF. Dense layer also checks loss + dL/dx;
        # Hub MoE cannot train so the MLA/MoE layer is forward-only.
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.create_pg(device)
        ckpt = self._ckpt_dir(require_hf_code=True)
        cfg = KimiLinearTinyConfig()
        layer_type = cfg.layers_type[layer_idx]
        is_moe = layer_idx >= cfg.first_k_dense_replace
        seq = 16

        with self.hf_impl():
            # from_hf unpacks fused_w1w3/fused_w2 from per-expert HF keys;
            # materialize_submodule only copies the first mapped tensor.
            model = self._build_xtuner(ckpt)
            xt_layer = model.layers[str(layer_idx)]

            hf = _load_hf_causal_lm(ckpt)
            hf_layer = hf.model.layers[layer_idx]
            hf_norm = hf.model.norm
            hf_lm_head = hf.lm_head

            # Fixed generator: dense layer asserts bitwise parity, so the input
            # must be identical across runs.
            g = torch.Generator(device="cuda").manual_seed(0)
            ids = torch.randint(0, 256, (1, seq), device="cuda", generator=g)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
            seq_ctx.to("cuda")
            dummy = torch.empty(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
            cos, sin = model.rotary_emb(dummy, seq_ctx.position_ids)
            labels = torch.randint(0, cfg.vocab_size, (seq,), device="cuda", generator=g)
            base = torch.randn(1, seq, cfg.hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
            block_residual = base.new_zeros(seq, 0, cfg.hidden_size)
            attn_mask = (
                None
                if layer_type == "linear_attention"
                else torch.triu(
                    torch.full((seq, seq), float("-inf"), device="cuda", dtype=torch.bfloat16),
                    diagonal=1,
                )[None, None]
            )
            position_ids = torch.arange(seq, device="cuda").unsqueeze(0)

            if is_moe:
                hf.eval()
                with torch.no_grad():
                    o_hf, _ = hf_layer(
                        base.clone(),
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        block_residual=block_residual.clone(),
                    )
                    o_xt = xt_layer(
                        base.clone(),
                        position_embeddings=(cos, sin),
                        seq_ctx=seq_ctx,
                        block_residual=block_residual.clone(),
                    )["hidden_states"]
                out_diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
                # KDA shares fla chunk_kda (bitwise). MLA has no XTUNER_HF_IMPL
                # eager path — XTuner flash_attn_varlen vs Hub eager softmax.
                layer_tol = 0.0 if layer_type == "linear_attention" else 3e-2
                self.assertLessEqual(
                    out_diff,
                    layer_tol,
                    f"layer {layer_idx} [{layer_type}] output max diff {out_diff} > {layer_tol}",
                )
                del hf
                dist.barrier()
                return

            hf.train()
            x_hf = base.clone().requires_grad_(True)
            o_hf, _ = hf_layer(
                x_hf,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
                block_residual=block_residual.clone(),
            )
            loss_hf = F.cross_entropy(hf_lm_head(hf_norm(o_hf)).reshape(-1, cfg.vocab_size), labels)
            loss_hf.backward()

            x_xt = base.clone().requires_grad_(True)
            o_xt = xt_layer(
                x_xt,
                position_embeddings=(cos, sin),
                seq_ctx=seq_ctx,
                block_residual=block_residual.clone(),
            )["hidden_states"]
            loss_xt = F.cross_entropy(
                F.linear(model.norm(o_xt), model.lm_head.weight).reshape(-1, cfg.vocab_size),
                labels,
            )
            loss_xt.backward()

        out_diff = (o_hf.float() - o_xt.float().reshape(o_hf.shape)).abs().max().item()
        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        grad_diff = (x_hf.grad.float() - x_xt.grad.float()).abs().max().item()  # type: ignore[union-attr]
        self.assertEqual(out_diff, 0.0, f"layer {layer_idx} [{layer_type}] output not bitwise: max diff {out_diff}")
        self.assertEqual(loss_diff, 0.0, f"layer {layer_idx} [{layer_type}] loss not bitwise: {loss_diff}")
        self.assertEqual(grad_diff, 0.0, f"layer {layer_idx} [{layer_type}] dL/dx not bitwise: max diff {grad_diff}")
        del hf
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_model_forward_parity(self, device):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.create_pg(device)
        ckpt = self._ckpt_dir(require_hf_code=True)
        vocab_size = KimiLinearTinyConfig().vocab_size
        input_ids = _text_to_ids(_PARITY_TEXTS[0], vocab_size, torch.device("cuda"))

        with self.hf_impl():
            hf = _load_hf_causal_lm(ckpt)
            with torch.no_grad():
                hf_out = hf(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
            loss_hf = hf_out.loss
            logits_hf = hf_out.logits
            del hf
            torch.cuda.empty_cache()

            xt = self._build_xtuner(ckpt)
            shift_input_ids = input_ids[:, :-1]
            shifted_labels = input_ids[:, 1:]
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))
            loss_cfg = CELossConfig()
            loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
            assert loss_ctx is not None
            loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
            with torch.no_grad():
                xt_out = xt(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            loss_xt = xt_out["loss"]
            logits_xt = xt_out["logits"]

        loss_diff = (loss_hf.float() - loss_xt.float()).abs().item()
        self.assertLessEqual(loss_diff, 1e-2, f"full-model loss diff {loss_diff}")
        hf_pred = logits_hf[:, :-1].argmax(dim=-1)
        xt_pred = logits_xt.argmax(dim=-1).reshape(hf_pred.shape)
        agree = (hf_pred == xt_pred).float().mean().item()
        self.assertGreaterEqual(agree, 0.9, f"argmax agreement {agree} too low")
        dist.barrier()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
        ],
    )
    def test_fsdp_accuracy(self, device, dispatcher, ep_size):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.create_pg(device)
        ckpt = self._ckpt_dir(require_hf_code=True)
        vocab_size = KimiLinearTinyConfig().vocab_size
        input_list = [_text_to_ids(text, vocab_size, torch.device("cuda")) for text in _PARITY_TEXTS]

        with self.hf_impl():
            hf_model = _load_hf_causal_lm(ckpt)
            expected_losses = []
            for input_ids in input_list:
                with torch.no_grad():
                    output = hf_model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
                expected_losses.append(output.loss)
            del hf_model
            torch.cuda.empty_cache()

            xt_model = self._build_xtuner(ckpt, dispatcher=dispatcher, ep_size=ep_size, fsdp=True)
            losses = [_xtuner_ce_loss(xt_model, input_ids) for input_ids in input_list]

        self._check_loss_curve(
            losses=torch.tensor(losses),
            losses_ref=torch.tensor(expected_losses),
            sim_tol=3e-2,
            rtol=3e-2,
        )
        dist.barrier()

    @parametrize.parametrize("device", [("cuda",)])
    def test_save_hf(self, device):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.create_pg(device)
        ckpt = self._ckpt_dir()
        origin_hf_path = Path(ckpt)

        # Save to shared storage: /tmp is container-local, so non-zero ranks
        # cannot see rank0's tmpdir on multi-node / multi-container setups.
        save_root = Path(self.temp_dir.name) / "save_hf_roundtrip"
        save_root.mkdir(parents=True, exist_ok=True)
        syncdir = [str(save_root)]
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast_object_list(syncdir, src=0)
        tmpdir_path = Path(syncdir[0])
        with self.hf_impl():
            xt_model = self._build_xtuner(ckpt, fsdp=True)
        xt_model.save_hf(tmpdir_path)

        origin_index_path = origin_hf_path / "model.safetensors.index.json"
        saved_index_path = tmpdir_path / "model.safetensors.index.json"

        if dist.get_rank() == 0:
            with open(tmpdir_path / "config.json") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["model_type"], "kimi_linear")
            self.assertEqual(saved_config["architectures"], ["KimiLinearForCausalLM"])
            self.assertTrue((tmpdir_path / "configuration_kimi_k3.py").is_file())
            self.assertTrue((tmpdir_path / "modeling_kimi_linear.py").is_file())

            with open(origin_index_path, "r") as f:
                origin_index = json.load(f)
            with open(saved_index_path, "r") as f:
                saved_index = json.load(f)

            # Origin tiny ckpt and FSDP save_hf both use `model-0001-save_rank0.safetensors`.
            # Cache by (kind, filename) so 8-rank shards are not aliased to the origin file.
            origin_fh: dict[str, object] = {}
            saved_fh: dict[str, object] = {}
            for key in origin_index["weight_map"].keys():
                origin_safetensor_name = origin_index["weight_map"][key]
                saved_safetensor_name = saved_index["weight_map"][key]
                if origin_safetensor_name not in origin_fh:
                    origin_fh[origin_safetensor_name] = safe_open(
                        str(origin_hf_path / origin_safetensor_name), framework="pt"
                    )
                if saved_safetensor_name not in saved_fh:
                    saved_fh[saved_safetensor_name] = safe_open(
                        str(tmpdir_path / saved_safetensor_name), framework="pt"
                    )
                origin_tensor = origin_fh[origin_safetensor_name].get_tensor(key)
                saved_tensor = saved_fh[saved_safetensor_name].get_tensor(key)
                self.assertTrue(torch.equal(origin_tensor, saved_tensor), f"tensor {key} is not equal")

            safetensor_keys: list[str] = []
            for fh in saved_fh.values():
                safetensor_keys.extend(fh.keys())
            safetensor_keys.sort()
            model_index_keys = sorted(saved_index["weight_map"].keys())
            self.assertListEqual(safetensor_keys, model_index_keys)

        dist.barrier()
        # Round-trip: the saved dir must load back (catches index/shard
        # mismatches that tensor equality alone misses).
        with self.hf_impl():
            model2 = self._build_xtuner(tmpdir_path, fsdp=True)
        del model2
        dist.barrier()


if __name__ == "__main__":
    unittest.main()
