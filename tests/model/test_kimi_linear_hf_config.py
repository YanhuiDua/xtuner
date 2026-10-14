import json
import os
from pathlib import Path

import pytest

import transformers
from xtuner._testing import HFConfigFieldDependency, check_hf_config_save
from xtuner.v1.model import get_model_config_from_hf


def _kimi_linear_hf_path() -> Path:
    configured = os.environ.get("KIMI_LINEAR_TINY_PATH")
    path = Path(configured) if configured else Path(__file__).parents[2] / "work_dirs/kimi_linear_tiny_bf16"
    if not (path / "config.json").is_file() or not (path / "configuration_kimi_k3.py").is_file():
        pytest.skip("set KIMI_LINEAR_TINY_PATH to a standalone KimiLinear HF checkpoint")
    return path


def test_save_hf_matches_transformers_and_engine_contracts():
    source = _kimi_linear_hf_path()
    config = get_model_config_from_hf(source)

    hf_config = config.hf_config
    assert hf_config is not None
    assert type(hf_config).__name__ == "KimiLinearConfig"
    assert type(hf_config).__module__.startswith("transformers_modules.")

    report = check_hf_config_save(
        config,
        source,
        trust_remote_code=True,
        engine_dependencies=(
            HFConfigFieldDependency(
                engine="vllm",
                version="0.31.0",
                path="/model_type",
                expected="kimi_linear",
                reason="vLLM selects its native KimiLinearConfig from model_type.",
                source=("https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/transformers_utils/config.py#L118"),
            ),
            HFConfigFieldDependency(
                engine="vllm",
                version="0.31.0",
                path="/architectures",
                expected=["KimiLinearForCausalLM"],
                reason="vLLM dispatches this architecture to its native Kimi-K3 text model.",
                source=(
                    "https://github.com/vllm-project/vllm/blob/v0.31.0/"
                    "vllm/model_executor/models/registry.py#L145-L148"
                ),
            ),
            HFConfigFieldDependency(
                engine="vllm",
                version="0.31.0",
                path="/moe_layer_freq",
                expected=1,
                reason="vLLM reads this field directly when selecting sparse decoder layers.",
                source=(
                    "https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/kimi_k3/nvidia/model.py#L867-L872"
                ),
            ),
            HFConfigFieldDependency(
                engine="sglang",
                version="0.5.21",
                path="/architectures",
                expected=["KimiLinearForCausalLM"],
                reason="SGLang uses this architecture to select its native KimiLinear entry class and MLA path.",
                source=(
                    "https://github.com/sgl-project/sglang/blob/v0.5.21/"
                    "python/sglang/srt/configs/model_config.py#L1275-L1287"
                ),
            ),
            HFConfigFieldDependency(
                engine="sglang",
                version="0.5.21",
                path="/moe_layer_freq",
                expected=1,
                reason="SGLang reads this field directly when selecting sparse decoder layers.",
                source=(
                    "https://github.com/sgl-project/sglang/blob/v0.5.21/"
                    "python/sglang/srt/models/kimi_linear.py#L586-L595"
                ),
            ),
        ),
    )

    assert report.transformers_version == transformers.__version__
    assert report.checked_engine_versions == ("vllm==0.31.0", "sglang==0.5.21")


def test_kimi_k3_exports_engine_compatible_text_config(tmp_path: Path):
    configured = os.environ.get("KIMI_K3_PATH")
    if not configured:
        pytest.skip("KIMI_K3_PATH is not set")

    source = Path(configured)
    if not (source / "configuration_kimi_k3.py").is_file():
        pytest.fail("KIMI_K3_PATH must contain configuration_kimi_k3.py for trust_remote_code")

    config = get_model_config_from_hf(source)
    config.save_hf(tmp_path)

    with open(tmp_path / "config.json", encoding="utf-8") as file:
        exported = json.load(file)

    assert exported["model_type"] == "kimi_linear"
    assert exported["architectures"] == ["KimiLinearForCausalLM"]
    assert exported["auto_map"] == {
        "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
        "AutoModel": "modeling_kimi_linear.KimiLinearModel",
        "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM",
    }
    assert exported["moe_layer_freq"] == 1
    assert exported["use_grouped_topk"] is True
    assert exported["linear_attn_config"]["full_attn_layers"][-1] == exported["num_hidden_layers"]
    assert (tmp_path / "configuration_kimi_k3.py").is_file()

    reloaded = transformers.AutoConfig.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True)
    assert type(reloaded).__name__ == "KimiLinearConfig"
