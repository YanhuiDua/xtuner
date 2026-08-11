from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

import xtuner.v1.train.trainer as trainer_module
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model import ConceptOlmoStage2V1Config
from xtuner.v1.model.dense.concept_olmo_flops import (
    ConceptOlmoFlopsEstimator,
    ConceptOlmoMFUCalculator,
    MFUConfig,
    get_peak_tflops_per_gpu,
)
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.train.trainer import Trainer


def _tiny_flops_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_position_embeddings=8,
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        concept_encoder_layers=2,
        concept_decoder_layers=2,
        concept_special_layers=1,
        concept_chunk_size=4,
        concept_v22_vq_codebook_size=2,
        concept_v22_vq_num_codebooks=2,
        window_size=(4, 0),
        window_attn_skip_freq=2,
    )


def test_concept_olmo_flops_exact_tiny_model():
    breakdown = ConceptOlmoFlopsEstimator(_tiny_flops_config()).estimate(
        sequence_length=8
    )

    assert breakdown.concept_sequence_length == 2
    assert breakdown.encoder_transformer_flops == 6176
    assert breakdown.encoder_route_flops == 848
    assert breakdown.concept_transformer_flops == 688
    assert breakdown.concept_route_flops == 116
    assert breakdown.concept_prediction_flops == 96
    assert breakdown.decoder_transformer_flops == 6176
    assert breakdown.decoder_route_flops == 1776
    assert breakdown.lm_head_flops == 1024
    assert breakdown.forward_flops == 16900
    assert breakdown.training_flops == 50700


def test_causal_and_sliding_attention_pair_counts():
    pair_count = ConceptOlmoFlopsEstimator._causal_attention_pairs

    assert pair_count(sequence_length=8, left_window=None) == 36
    assert pair_count(sequence_length=8, left_window=4) == 30
    assert pair_count(sequence_length=4, left_window=4) == 10


def test_stage2_v1_mfu_uses_global_step_flops():
    config = ConceptOlmoStage2V1Config()
    calculator = ConceptOlmoMFUCalculator(
        model_config=config,
        mfu_config=MFUConfig(sequence_length=8192),
        global_batch_size=16,
        world_size=8,
        device_name="NVIDIA H200",
    )

    assert calculator.breakdown.concept_sequence_length == 2048
    assert calculator.breakdown.training_flops == 402_612_968_669_184
    assert calculator.model_flops_per_step == 6_441_807_498_706_944

    metrics = calculator.compute(step_time=4.0)
    assert metrics.tflops_per_gpu == pytest.approx(
        calculator.model_flops_per_step / 4.0 / 8 / 1e12
    )
    assert calculator.peak_tflops_per_gpu == 989.0
    assert calculator.peak_tflops_source == "auto"
    assert metrics.mfu == pytest.approx(metrics.tflops_per_gpu / 989.0)


@pytest.mark.parametrize(
    ("device_name", "expected_peak"),
    [
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA H100 PCIe", 756.0),
        ("NVIDIA H100 NVL", 835.0),
        ("NVIDIA H100 80GB HBM3", 989.0),
        ("NVIDIA H200", 989.0),
        ("NVIDIA GB200", 2500.0),
        ("NVIDIA B200", 2250.0),
        ("NVIDIA L40S", 362.0),
    ],
)
def test_peak_tflops_are_inferred_from_device_name(device_name, expected_peak):
    assert get_peak_tflops_per_gpu(device_name) == expected_peak


def test_peak_tflops_override_and_unknown_device():
    calculator = ConceptOlmoMFUCalculator(
        model_config=_tiny_flops_config(),
        mfu_config=MFUConfig(sequence_length=8, peak_tflops_per_gpu=100.0),
        global_batch_size=1,
        world_size=1,
    )
    assert calculator.peak_tflops_per_gpu == 100.0
    assert calculator.peak_tflops_source == "configured"

    with pytest.raises(ValueError, match="peak_tflops_per_gpu explicitly"):
        get_peak_tflops_per_gpu("Unknown Accelerator")


def test_trainer_builds_mfu_calculator_from_runtime_device(monkeypatch):
    fake_device_module = SimpleNamespace(get_device_name=lambda: "NVIDIA H200")
    monkeypatch.setattr(trainer_module, "DEVICE_MODULE", fake_device_module)
    monkeypatch.setattr(trainer_module, "get_world_size", lambda: 8)

    trainer = object.__new__(Trainer)
    trainer._global_batch_size = 16
    trainer._fsdp_config = FSDPConfig()
    calculator = trainer._build_mfu_calculator(
        ConceptOlmoStage2V1Config(),
        MFUConfig(sequence_length=8192),
    )

    assert calculator is not None
    assert calculator.device_name == "NVIDIA H200"
    assert calculator.peak_tflops_per_gpu == 989.0
    assert calculator.peak_tflops_source == "auto"

    trainer._fsdp_config = FSDPConfig(param_dtype=torch.float32)
    with pytest.raises(ValueError, match="supports BF16/FP16"):
        trainer._build_mfu_calculator(
            ConceptOlmoStage2V1Config(),
            MFUConfig(sequence_length=8192),
        )


def test_flops_and_mfu_validate_inputs():
    estimator = ConceptOlmoFlopsEstimator(_tiny_flops_config())
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        estimator.estimate(sequence_length=9)
    with pytest.raises(ValidationError):
        MFUConfig(sequence_length=0, peak_tflops_per_gpu=1.0)
    with pytest.raises(ValidationError):
        MFUConfig(sequence_length=1, peak_tflops_per_gpu=0.0)
    assert MFUConfig(sequence_length=1).peak_tflops_per_gpu is None


def test_trainer_step_log_and_tracker_include_mfu(monkeypatch):
    fake_device_module = SimpleNamespace(
        max_memory_allocated=lambda: 2 * 1024**3,
        max_memory_reserved=lambda: 3 * 1024**3,
        reset_peak_memory_stats=lambda: None,
    )
    monkeypatch.setattr(trainer_module, "DEVICE_MODULE", fake_device_module)

    trainer = object.__new__(Trainer)
    trainer._cur_epoch = 1
    trainer._cur_step = 3
    trainer._total_step = 10
    trainer._train_time = 12.0
    trainer._lr_scheduler = Mock()
    trainer._lr_scheduler.get_last_lr.return_value = [1e-5]
    trainer.logger = Mock()
    trainer._exp_tracker = Mock()

    trainer._log_step(
        train_step_info={
            "total_loss": 0.9,
            "step_consumed_tokens": 8192,
            "step_consumed_img_tokens": 0.0,
            "efficient_attn_ratio": 0.5,
            "img_efficient_attn_ratio": 0.0,
            "logs_info": {"reduced_llm_loss": 0.8},
            "extra_info": ModelForwardExtraLogInfo(),
        },
        training_metrics={
            "local_step_consumed_tokens": 8192,
            "local_step_consumed_img_tokens": 0,
            "local_total_consumed_tokens": 24576,
            "approximate_total_consumed_tokens": 196608,
            "tgs": 2048.0,
            "exp_tgs": 2048.0,
            "eta_seconds": 28.0,
            "eta_hms": "0:00:28",
            "e2e_train_time": 12.0,
            "model_flops_per_step": 8_000_000_000_000_000,
            "tflops_per_gpu": 250.0,
            "mfu": 0.25,
        },
        grad_norm=1.0,
        data_time=0.1,
        step_time=4.0,
    )

    message = trainer.logger.info.call_args.args[0]
    assert "model_flops_per_step: 8.000 PF" in message
    assert "tflops_per_gpu: 250.0" in message
    assert "mfu: 25.00%" in message

    scalars = trainer._exp_tracker.add_scalars.call_args.kwargs["tag_scalar_dict"]
    assert scalars["runtime_info/model_flops_per_step"] == 8_000_000_000_000_000
    assert scalars["runtime_info/tflops_per_gpu"] == 250.0
    assert scalars["runtime_info/mfu"] == 0.25
