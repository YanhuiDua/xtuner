import json
import os
from pathlib import Path

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model import (
    ConceptLMV22VQConfig,
    ConceptOlmoStage2V1Config,
    get_model_config,
    get_model_config_from_hf,
)
from xtuner.v1.module import MHAConfig
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


def _initialize(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)


def test_config_alias_and_meta_layout():
    config = get_model_config("concept-olmo-stage2-v1")
    assert isinstance(config, ConceptOlmoStage2V1Config)
    with torch.device("meta"):
        model = config.build()
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_938_363_793
    assert len(model.state_dict()) == 724
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


@pytest.mark.skipif(MODEL_PATH is None, reason="CONCEPT_OLMO_STAGE2_V1_PATH is not set")
def test_checkpoint_config_and_key_coverage():
    assert MODEL_PATH is not None
    path = Path(MODEL_PATH)
    config = ConceptLMV22VQConfig.from_hf(path)
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
    assert len(loaded) == 724
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
