import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed.nn.functional as distF
from torch.distributed.device_mesh import init_device_mesh

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model import (
    ConceptLMV22VQConfig,
    ConceptOlmoStage2V1Config,
    get_model_config,
    get_model_config_from_hf,
)
from xtuner.v1.module import ConceptStage, DecoderStage, EncoderStage, MHAConfig
from xtuner.v1.module.rope import RopeParametersConfig


MODEL_PATH = os.getenv("CONCEPT_OLMO_STAGE2_V1_PATH")


def _tiny_config() -> ConceptLMV22VQConfig:
    return ConceptLMV22VQConfig(
        vocab_size=128,
        max_position_embeddings=32,
        eos_token_id=1,
        bos_token_id=1,
        pad_token_id=0,
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            qk_norm=True,
            rms_norm_eps=1e-6,
            sliding_window=4,
            attn_impl="eager_attention",
        ),
        rope_parameters_cfg=RopeParametersConfig(rope_theta=500000.0),
        concept_encoder_layers=2,
        concept_decoder_layers=2,
        concept_special_layers=2,
        concept_chunk_size=4,
        concept_shift_feature=True,
        concept_chunk_merge_method="meanpooling",
        concept_v22_vq_codebook_size=16,
        concept_v22_vq_num_codebooks=4,
        concept_fusion_norm_alpha_init=0.1,
        concept_dd_two_route_add_decoder_use_softmax=True,
        window_size=(4, 0),
        window_attn_skip_freq=2,
        compile_cfg=False,
    )


def _fp8_tiny_config() -> ConceptLMV22VQConfig:
    return _tiny_config().model_copy(
        update={
            "hidden_size": 128,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "concept_v22_vq_codebook_size": 128,
            "attention": MHAConfig(
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=32,
                qk_norm=True,
                rms_norm_eps=1e-6,
                sliding_window=4,
                attn_impl="flash_attention",
            ),
            "float8_cfg": Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TENSORWISE,
            ),
            "compile_cfg": False,
        }
    )


def _initialize(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)


def test_config_alias_and_meta_layout():
    config = get_model_config("concept-olmo-stage2-v1")
    assert isinstance(config, ConceptOlmoStage2V1Config)
    config.compile_cfg = False
    with torch.device("meta"):
        model = config.build()
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_938_363_793
    assert len(model.state_dict()) == 693
    assert model.concept_quantizer.codebook.shape == (32 * 128, 128)
    assert model.concept_quantizer.transformed_codebook().shape == (32, 128, 128)
    assert model.to_hf_key_list("concept_quantizer.codebook") == [
        f"concept_quantizer.codebook.{index}" for index in range(32)
    ]
    assert model.lm_head.weight.shape == (config.vocab_size, 4096)
    assert model.lm_head.logical_out_features == config.vocab_size
    assert model.to_hf_key_list("encoder_stages.3.transformer.self_attention.linear_qkv.weight") == [
        "encoder.layers.3.self_attention.linear_qkv.weight"
    ]
    assert model.to_hf_key_list("decoder_stages.2.read_concept_route.w2.weight") == [
        "decoder_read_concept_routes.2.w2.weight"
    ]
    assert {key for key in model.state_dict() if key.endswith("_extra_state")} == {
        "concept_final_norm._extra_state",
        "final_norm._extra_state",
    }


def test_lm_head_vocab_padding_checkpoint_conversion():
    config = _tiny_config().model_copy(
        update={"vocab_size": 126, "lm_head_vocab_padding_multiple": 8}
    )
    model = config.build()
    checkpoint_weight = torch.randn(config.vocab_size, config.hidden_size)
    local_weight = torch.empty(128, config.hidden_size)

    model.safetensors_to_params(
        [checkpoint_weight],
        local_weight,
        param_name="lm_head.weight",
        start=None,
        end=None,
        dim=None,
    )
    torch.testing.assert_close(local_weight[: config.vocab_size], checkpoint_weight)
    torch.testing.assert_close(local_weight[config.vocab_size :], torch.zeros_like(local_weight[-2:]))
    saved_weight = model.param_to_safetensor(local_weight, "output_layer.weight")
    torch.testing.assert_close(saved_weight, checkpoint_weight)

    local_shard = torch.empty(16, config.hidden_size)
    model.safetensors_to_params(
        [checkpoint_weight],
        local_shard,
        param_name="lm_head.weight",
        start=112,
        end=128,
        dim=None,
    )
    torch.testing.assert_close(local_shard[:14], checkpoint_weight[112:126])
    torch.testing.assert_close(local_shard[14:], torch.zeros_like(local_shard[14:]))


def test_packed_documents_match_separate_forwards(monkeypatch):
    monkeypatch.setenv("XTUNER_HF_IMPL", "true")
    model = _tiny_config().build().eval()
    _initialize(model)
    document_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    document_b = torch.tensor([[8, 9, 10, 11, 12]])

    packed_ctx = SequenceContext.from_input_ids((document_a, document_b), device="cpu")
    separate_contexts = [
        SequenceContext.from_input_ids((document,), device="cpu") for document in (document_a, document_b)
    ]
    with torch.no_grad():
        packed_logits = model(packed_ctx).logits
        separate_logits = torch.cat([model(context).logits for context in separate_contexts], dim=1)
    torch.testing.assert_close(packed_logits, separate_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FSDP")
class TestConceptOlmoFSDP(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_forward_backward_with_recompute(self):
        self.create_pg("cuda")
        config = _tiny_config()
        baseline = config.build().cuda().eval()
        _initialize(baseline)
        state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
        with torch.no_grad():
            expected = baseline(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
        del baseline

        model = config.build().cuda()
        model.load_state_dict(state)
        model.fully_shard(
            FSDPConfig(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                recompute_ratio=1.0,
            )
        )
        previous_offload = os.environ.get("XTUNER_ACTIVATION_OFFLOAD")
        os.environ["XTUNER_ACTIVATION_OFFLOAD"] = "1"
        try:
            output = model(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
            output.sum().backward()
            assert any(parameter.grad is not None for parameter in model.parameters())
        finally:
            if previous_offload is None:
                os.environ.pop("XTUNER_ACTIVATION_OFFLOAD", None)
            else:
                os.environ["XTUNER_ACTIVATION_OFFLOAD"] = previous_offload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for optimization tests")
class TestConceptOlmoOptimizations(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def test_torch_compile_forward_backward(self):
        self.create_pg("cuda")
        config = _tiny_config()
        config.compile_cfg = None
        config.attention.attn_impl = "flash_attention"
        model = config.build().cuda().bfloat16()
        _initialize(model)
        input_ids = torch.arange(1, 9, device="cuda").unsqueeze(0)
        output = model(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
        output.float().sum().backward()
        assert output.shape == (1, 8, config.vocab_size)
        for stage_type in (EncoderStage, ConceptStage, DecoderStage):
            assert hasattr(stage_type.forward, "_torchdynamo_orig_callable")

    def test_tensorwise_fp8_forward_backward(self):
        self.create_pg("cuda")
        config = _fp8_tiny_config()
        baseline_config = config.model_copy(update={"float8_cfg": None})
        baseline = baseline_config.build().cuda().bfloat16()
        _initialize(baseline)
        state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
        input_ids = (torch.arange(64, device="cuda") % config.vocab_size).unsqueeze(0)
        with torch.no_grad():
            expected = baseline(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits.float()
        del baseline

        model = config.build().cuda().bfloat16()
        model.load_state_dict(state)
        model.fully_shard(FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        assert model.float8_handler is not None
        model.float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)
        actual = model(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
        actual.float().sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required for SP")
class TestConceptOlmoSequenceParallel(DeterministicDDPTestCase):

    @property
    def world_size(self) -> int:
        return 2

    def test_packed_forward_backward(self):
        self.create_pg("cuda")
        sp_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("sp",))
        with self.hf_impl():
            model = _tiny_config().build().cuda().eval()
            _initialize(model)
            document_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
            document_b = torch.tensor([[10, 11, 12, 13, 14]], device="cuda")

            full_ctx = SequenceContext.from_input_ids((document_a, document_b), device="cuda")
            with torch.no_grad():
                expected = model(full_ctx).logits

            sp_ctx = SequenceContext.from_input_ids((document_a, document_b), device="cuda").split(sp_mesh)
            actual_shard = model(sp_ctx).logits
            actual = torch.cat(
                distF.all_gather(actual_shard.contiguous(), group=sp_mesh.get_group()),
                dim=1,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            actual_shard.sum().backward()
            assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(MODEL_PATH is None, reason="CONCEPT_OLMO_STAGE2_V1_PATH is not set")
def test_checkpoint_config_and_key_coverage():
    assert MODEL_PATH is not None
    path = Path(MODEL_PATH)
    config = ConceptLMV22VQConfig.from_hf(path)
    config.compile_cfg = False
    assert config.rope_theta == 500000.0
    assert config.window_size == (4096, 0)
    assert isinstance(get_model_config_from_hf(path), ConceptLMV22VQConfig)
    with torch.device("meta"):
        model = config.build()
    with open(path / "model.safetensors.index.json") as file:
        checkpoint_keys = set(json.load(file)["weight_map"])
    mapped_keys = {hf_key for key in model.state_dict() for hf_key in model.to_hf_key_list(key)}
    assert mapped_keys == checkpoint_keys


@pytest.mark.skipif(MODEL_PATH is None, reason="CONCEPT_OLMO_STAGE2_V1_PATH is not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the full checkpoint")
def test_full_checkpoint_forward_bitwise(monkeypatch):
    assert MODEL_PATH is not None
    monkeypatch.setenv("XTUNER_HF_IMPL", "true")
    config = ConceptLMV22VQConfig.from_hf(MODEL_PATH)
    config.compile_cfg = False
    with torch.device("meta"):
        model = config.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
    loaded, unloaded, missing = model.from_hf(MODEL_PATH)
    assert len(loaded) == 693
    assert not unloaded
    assert not missing

    input_ids = torch.arange(1, 9, device="cuda", dtype=torch.long).unsqueeze(0)
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
    with torch.no_grad():
        logits = model(seq_ctx).logits
    expected = torch.tensor(
        [4.90625, -6.21875, 2.796875, -1.8515625, -2.78125, 0.185546875, -3.796875, -6.03125]
    )
    torch.testing.assert_close(logits[0, -1, :8].cpu(), expected, rtol=0, atol=0)
