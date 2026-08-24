import gc
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed.nn.functional as distF
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import AutoModelForCausalLM, AutoTokenizer
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import (
    ConceptLMV22VQ,
    ConceptLMV22VQConfig,
    ConceptOlmoStage2V1Config,
    get_model_config,
    get_model_config_from_hf,
)
from xtuner.v1.module import ConceptStage, DecoderStage, DepthDD, EncoderStage, MHAConfig, ResidualRoute
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.utils.load_spec import LoadEnum


MODEL_PATH = os.getenv("CONCEPT_OLMO_STAGE2_V1_PATH")
ALIGNMENT_PROMPTS = (
    "Explain gravity in one concise sentence.",
    "Write a Python function that adds two integers, and explain how it "
    "handles negative inputs.",
    "请用三句话介绍机器学习中的过拟合，并给出一种常见的缓解方法。",
)


def _global_grad_norms(model: torch.nn.Module) -> tuple[float, float, int]:
    """Return FP32 two-level and FP64 diagnostic global L2 norms."""
    parameter_norms_fp32: list[torch.Tensor] = []
    parameter_norms_fp64: list[torch.Tensor] = []
    grad_numel = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter_norms_fp32.append(torch.linalg.vector_norm(parameter.grad.detach(), ord=2.0, dtype=torch.float32))
        parameter_norms_fp64.append(torch.linalg.vector_norm(parameter.grad.detach(), ord=2.0, dtype=torch.float64))
        grad_numel += parameter.grad.numel()
    if not parameter_norms_fp32:
        raise AssertionError("Expected at least one parameter gradient")
    total_norm_fp32 = torch.linalg.vector_norm(torch.stack(parameter_norms_fp32), ord=2.0, dtype=torch.float32)
    total_norm_fp64 = torch.linalg.vector_norm(torch.stack(parameter_norms_fp64), ord=2.0, dtype=torch.float64)
    return total_norm_fp32.item(), total_norm_fp64.item(), grad_numel


def _tensor_error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    absolute_error = (actual_fp32 - expected_fp32).abs()
    relative_error = absolute_error / expected_fp32.abs().clamp_min(1e-6)
    return {
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs_error": absolute_error.max().item(),
        "mean_abs_error": absolute_error.mean().item(),
        "max_rel_error": relative_error.max().item(),
    }


def _build_alignment_prompt_batch(
    model_path: str | Path,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], torch.Tensor, torch.Tensor, list[int]]:
    """Tokenize the shared three-prompt varlen alignment batch."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    documents_cpu = tuple(
        tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )
        for prompt in ALIGNMENT_PROMPTS
    )
    document_lengths = tuple(document.shape[1] for document in documents_cpu)
    assert all(document.shape[0] == 1 and document.shape[1] > 1 for document in documents_cpu)
    assert len(set(document_lengths)) == len(documents_cpu), (
        "The prompts must exercise genuinely varlen input"
    )
    if tokenizer.pad_token_id is not None:
        assert all(not torch.eq(document, tokenizer.pad_token_id).any() for document in documents_cpu)

    shifted_label_documents: list[torch.Tensor] = []
    for document in documents_cpu:
        labels = torch.full_like(document, -100)
        labels[:, :-1] = document[:, 1:]
        shifted_label_documents.append(labels)
    shifted_labels = torch.cat(shifted_label_documents, dim=1).to(device)
    loss_weight = (shifted_labels != -100).float()
    loss_weight /= loss_weight.sum()
    assert int((shifted_labels != -100).sum()) == sum(length - 1 for length in document_lengths)

    cu_seq_lens = [0]
    for document_length in document_lengths:
        cu_seq_lens.append(cu_seq_lens[-1] + document_length)
    return documents_cpu, document_lengths, shifted_labels, loss_weight, cu_seq_lens


def _assert_alignment_sequence_context(
    seq_ctx: SequenceContext,
    document_lengths: tuple[int, ...],
    expected_cu_seq_lens: list[int],
) -> None:
    assert seq_ctx.cu_seq_lens_q.cpu().tolist() == expected_cu_seq_lens
    assert seq_ctx.position_ids is not None
    position_chunks = seq_ctx.position_ids.cpu().split(document_lengths, dim=1)
    for positions, document_length in zip(position_chunks, document_lengths, strict=True):
        torch.testing.assert_close(positions, torch.arange(document_length).unsqueeze(0), rtol=0, atol=0)


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


def _stage_group_tiny_config(stage_group_size: int) -> ConceptLMV22VQConfig:
    return _tiny_config().model_copy(
        update={
            "num_hidden_layers": 10,
            "concept_encoder_layers": 5,
            "concept_decoder_layers": 5,
            "concept_special_layers": 5,
            "stage_group_size": stage_group_size,
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
    assert len(model.state_dict()) == 631
    assert model.concept_quantizer.codebook.shape == (32 * 128, 128)
    assert model.concept_quantizer.transformed_codebook().shape == (32, 128, 128)
    assert model.to_hf_key_list("concept_quantizer.codebook") == [
        f"concept_quantizer.codebook.{index}" for index in range(32)
    ]
    assert model.prediction_heads.weight.shape == (32 * 128, 4096)
    assert model.prediction_heads.bias.shape == (32 * 128,)
    assert model.to_hf_key_list("prediction_heads.weight") == [
        f"concept_predictor.prediction_heads.{index}.weight" for index in range(32)
    ]
    assert model.to_hf_key_list("prediction_heads.bias") == [
        f"concept_predictor.prediction_heads.{index}.bias" for index in range(32)
    ]
    assert model.load_spec_mapping["prediction_heads.weight"].load_enum is LoadEnum.FUSED
    assert model.load_spec_mapping["prediction_heads.weight"].dim == 0
    assert model.load_spec_mapping["prediction_heads.bias"].load_enum is LoadEnum.FUSED
    assert model.load_spec_mapping["prediction_heads.bias"].dim == 0
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

    grouped_config = config.model_copy(update={"stage_group_size": 4})
    with torch.device("meta"):
        grouped_model = grouped_config.build()
    assert len(grouped_model.state_dict()) == 631
    assert [len(group) for group in grouped_model.encoder_stage_groups] == [4, 4, 4, 4]
    assert [len(group) for group in grouped_model.concept_stage_groups] == [4, 4]
    assert [len(group) for group in grouped_model.decoder_stage_groups] == [4, 4, 4, 4]
    assert grouped_model.to_hf_key_list(
        "encoder_stage_groups.0.3.transformer.self_attention.linear_qkv.weight"
    ) == ["encoder.layers.3.self_attention.linear_qkv.weight"]
    assert grouped_model.to_hf_key_list("decoder_stage_groups.0.2.read_concept_route.w2.weight") == [
        "decoder_read_concept_routes.2.w2.weight"
    ]
    grouped_hf_keys = sorted(
        hf_key for key in grouped_model.state_dict() for hf_key in grouped_model.to_hf_key_list(key)
    )
    ungrouped_hf_keys = sorted(hf_key for key in model.state_dict() for hf_key in model.to_hf_key_list(key))
    assert grouped_hf_keys == ungrouped_hf_keys


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


def test_fused_prediction_heads_match_independent_heads():
    config = _tiny_config()
    model = config.build()
    fused_head = model.prediction_heads
    num_codebooks = config.concept_v22_vq_num_codebooks
    codebook_size = config.concept_v22_vq_codebook_size

    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        fused_head.weight.copy_(torch.randn(fused_head.weight.shape, generator=generator))
        fused_head.bias.copy_(torch.randn(fused_head.bias.shape, generator=generator))

    fused_input = torch.randn(2, 3, config.hidden_size, generator=generator, requires_grad=True)
    reference_input = fused_input.detach().clone().requires_grad_()
    reference_weights = [chunk.detach().clone().requires_grad_() for chunk in fused_head.weight.chunk(num_codebooks)]
    reference_biases = [chunk.detach().clone().requires_grad_() for chunk in fused_head.bias.chunk(num_codebooks)]

    actual = fused_head(fused_input).reshape(2, 3, num_codebooks, codebook_size)
    expected = torch.stack(
        [
            F.linear(reference_input, weight, bias)
            for weight, bias in zip(reference_weights, reference_biases, strict=True)
        ],
        dim=2,
    )
    torch.testing.assert_close(actual, expected)

    output_grad = torch.randn(actual.shape, generator=generator)
    actual.backward(output_grad)
    expected.backward(output_grad)
    torch.testing.assert_close(fused_input.grad, reference_input.grad)
    torch.testing.assert_close(fused_head.weight.grad, torch.cat([weight.grad for weight in reference_weights]))
    torch.testing.assert_close(fused_head.bias.grad, torch.cat([bias.grad for bias in reference_biases]))


def test_fused_prediction_heads_checkpoint_conversion():
    config = _tiny_config()
    model = config.build()
    num_codebooks = config.concept_v22_vq_num_codebooks
    codebook_size = config.concept_v22_vq_codebook_size
    weights = [torch.full((codebook_size, config.hidden_size), float(index)) for index in range(num_codebooks)]
    biases = [torch.full((codebook_size,), float(index)) for index in range(num_codebooks)]
    fused_weight = torch.empty(num_codebooks * codebook_size, config.hidden_size)
    fused_bias = torch.empty(num_codebooks * codebook_size)

    model.safetensors_to_params(
        weights,
        fused_weight,
        param_name="prediction_heads.weight",
        start=None,
        end=None,
        dim=0,
    )
    model.safetensors_to_params(
        biases,
        fused_bias,
        param_name="prediction_heads.bias",
        start=None,
        end=None,
        dim=0,
    )
    torch.testing.assert_close(fused_weight, torch.cat(weights))
    torch.testing.assert_close(fused_bias, torch.cat(biases))


@pytest.mark.parametrize("use_softmax", [False, True])
def test_depth_dd_tuple_history_matches_materialized_reference(monkeypatch, use_softmax):
    hidden_size = 8
    layer_idx = 3
    generator = torch.Generator().manual_seed(19)
    depth_dd = DepthDD(hidden_size, layer_idx, use_softmax).double()
    _initialize(depth_dd)

    history = tuple(
        torch.randn(2, 3, hidden_size, generator=generator, dtype=torch.float64, requires_grad=True)
        for _ in range(layer_idx + 2)
    )
    hidden_states = history[-1]
    weights = depth_dd.w2(F.gelu(depth_dd.w1(hidden_states)))
    weights = weights + depth_dd.static_a.view(1, 1, -1)
    if use_softmax:
        weights = weights.softmax(dim=-1)
    expected = (weights.unsqueeze(-1) * torch.stack(history, dim=2)).sum(dim=2)

    # Exercise the compiled-path reduction; ordinary eager execution keeps the
    # materialized reference path to preserve HuggingFace reduction order.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    actual = depth_dd(hidden_states, history)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    output_grad = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_grads = torch.autograd.grad(actual, (*history, *depth_dd.parameters()), output_grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (*history, *depth_dd.parameters()), output_grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("num_source_states", [8, 16])
@pytest.mark.parametrize("use_softmax", [False, True])
def test_residual_route_compiled_reduction_matches_einsum(monkeypatch, num_source_states, use_softmax):
    hidden_size = 8
    generator = torch.Generator().manual_seed(23)
    route = ResidualRoute(hidden_size, num_source_states, use_softmax).double()
    _initialize(route)

    target_hidden = torch.randn(2, 3, hidden_size, generator=generator, dtype=torch.float64, requires_grad=True)
    source_states = torch.randn(
        2,
        3,
        num_source_states,
        hidden_size,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    residual_scale = torch.randn((), generator=generator, dtype=torch.float64, requires_grad=True)

    weights = route.w2(F.gelu(route.w1(target_hidden)))
    if use_softmax:
        weights = weights.softmax(dim=-1)
    source_mix = torch.einsum("bsm,bsmh->bsh", weights, source_states)
    expected = target_hidden + source_mix * route.residual_diag * residual_scale

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    actual = route(target_hidden, source_states, residual_scale)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    output_grad = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    grad_inputs = (target_hidden, source_states, residual_scale, *route.parameters())
    actual_grads = torch.autograd.grad(actual, grad_inputs, output_grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, grad_inputs, output_grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


def test_stage_group_eager_forward_backward_matches_individual_stages():
    baseline = _stage_group_tiny_config(stage_group_size=1).build().eval()
    grouped = _stage_group_tiny_config(stage_group_size=4).build().eval()
    _initialize(baseline)

    assert [len(group) for group in grouped.encoder_stage_groups] == [4, 1]
    assert [len(group) for group in grouped.concept_stage_groups] == [4, 1]
    assert [len(group) for group in grouped.decoder_stage_groups] == [4, 1]
    assert grouped.to_hf_key_list(
        "encoder_stage_groups.1.4.transformer.self_attention.linear_qkv.weight"
    ) == ["encoder.layers.4.self_attention.linear_qkv.weight"]

    baseline_parameters = list(baseline.named_parameters())
    grouped_parameters = list(grouped.named_parameters())
    assert len(grouped_parameters) == len(baseline_parameters)
    with torch.no_grad():
        for (baseline_name, baseline_parameter), (grouped_name, grouped_parameter) in zip(
            baseline_parameters, grouped_parameters, strict=True
        ):
            assert grouped.to_hf_key_list(grouped_name) == baseline.to_hf_key_list(baseline_name)
            grouped_parameter.copy_(baseline_parameter)

    input_ids = torch.arange(1, 9).unsqueeze(0)
    baseline_logits = baseline(SequenceContext.from_input_ids((input_ids,), device="cpu")).logits
    grouped_logits = grouped(SequenceContext.from_input_ids((input_ids,), device="cpu")).logits
    torch.testing.assert_close(grouped_logits, baseline_logits, rtol=1e-6, atol=1e-6)

    generator = torch.Generator().manual_seed(29)
    output_grad = torch.randn(grouped_logits.shape, generator=generator)
    baseline_logits.backward(output_grad)
    grouped_logits.backward(output_grad)
    for (_, baseline_parameter), (_, grouped_parameter) in zip(
        baseline_parameters, grouped_parameters, strict=True
    ):
        assert (grouped_parameter.grad is None) == (baseline_parameter.grad is None)
        if baseline_parameter.grad is not None:
            torch.testing.assert_close(grouped_parameter.grad, baseline_parameter.grad, rtol=1e-5, atol=1e-6)


def test_lengths_single_sequence_uses_cpu_max_length():
    seq_ctx = SequenceContext.from_input_ids((torch.arange(8).unsqueeze(0),), device="cpu")
    assert seq_ctx.max_length_q.device.type == "cpu"
    assert seq_ctx.num_padding == 0

    # A distinct cumulative length proves this branch reads the CPU maximum
    # instead of materializing cu_seq_lens_q values.
    seq_ctx.cu_seq_lens_q = torch.tensor([0, 99], dtype=torch.int32)
    assert ConceptLMV22VQ._lengths(seq_ctx) == [8]


def test_lengths_packed_sequence_uses_cumulative_lengths():
    documents = (torch.arange(3).unsqueeze(0), torch.arange(5).unsqueeze(0))
    seq_ctx = SequenceContext.from_input_ids(documents, device="cpu")
    seq_ctx.max_length_q = torch.tensor(99)

    assert ConceptLMV22VQ._lengths(seq_ctx) == [3, 5]


def test_lengths_padded_sequence_uses_cumulative_lengths():
    seq_ctx = SequenceContext.from_input_ids((torch.arange(8).unsqueeze(0),), device="cpu")
    seq_ctx.num_padding = 2
    seq_ctx.max_length_q = torch.tensor(6)

    assert ConceptLMV22VQ._lengths(seq_ctx) == [8]


@pytest.mark.parametrize("token_lengths", [[9], [7, 10]], ids=["single_document", "packed_documents"])
@pytest.mark.parametrize("shift_feature", [False, True])
def test_concept_document_transforms_match_cat_reference(token_lengths, shift_feature):
    config = _tiny_config().model_copy(update={"concept_shift_feature": shift_feature})
    model = config.build()
    generator = torch.Generator().manual_seed(37)
    hidden_states = torch.randn(1, sum(token_lengths), config.hidden_size, generator=generator)

    merged_chunks = []
    start = 0
    for seq_len in token_lengths:
        state = hidden_states[:, start : start + seq_len]
        start += seq_len
        if seq_len < config.concept_chunk_size:
            merged = state.mean(dim=1, keepdim=True)
        else:
            usable = (seq_len // config.concept_chunk_size) * config.concept_chunk_size
            merged = state[:, :usable].reshape(
                1,
                usable // config.concept_chunk_size,
                config.concept_chunk_size,
                config.hidden_size,
            ).mean(dim=2)
        merged_chunks.append(merged)
    expected_merged = torch.cat(merged_chunks, dim=1)
    expected_concept_lengths = [chunk.shape[1] for chunk in merged_chunks]
    actual_merged, actual_concept_lengths = model._merge_chunks(hidden_states, token_lengths)
    torch.testing.assert_close(actual_merged, expected_merged)
    assert actual_concept_lengths == expected_concept_lengths

    def repeat_shift_reference(
        states: torch.Tensor,
        state_lengths: list[int],
        prepend_zero: bool,
    ) -> torch.Tensor:
        repeated_documents = []
        state_start = 0
        for state_len, seq_len in zip(state_lengths, token_lengths, strict=True):
            state = states[:, state_start : state_start + state_len]
            state_start += state_len
            if prepend_zero:
                state = torch.cat((torch.zeros_like(state[:, :1]), state), dim=1)
            repeated = state.repeat_interleave(config.concept_chunk_size, dim=1)[:, : seq_len + 1]
            repeated = repeated[:, 1:] if shift_feature else repeated[:, :seq_len]
            if repeated.shape[1] < seq_len:
                padding = repeated.new_zeros(1, seq_len - repeated.shape[1], *repeated.shape[2:])
                repeated = torch.cat((repeated, padding), dim=1)
            repeated_documents.append(repeated[:, :seq_len])
        return torch.cat(repeated_documents, dim=1)

    expected_repeated = repeat_shift_reference(expected_merged, expected_concept_lengths, True)
    actual_repeated = model._repeat_shift(
        actual_merged,
        actual_concept_lengths,
        token_lengths,
    )
    torch.testing.assert_close(actual_repeated, expected_repeated)

    source_states = torch.randn(
        1,
        sum(expected_concept_lengths),
        3,
        config.hidden_size,
        generator=generator,
    )
    source_chunks = []
    start = 0
    for concept_len in expected_concept_lengths:
        state = source_states[:, start : start + concept_len]
        start += concept_len
        source_chunks.append(torch.cat((torch.zeros_like(state[:, :1]), state), dim=1))
    expected_sources = torch.cat(source_chunks, dim=1)
    expected_source_lengths = [length + 1 for length in expected_concept_lengths]
    actual_sources, actual_source_lengths = model._prepend_zero_chunks(
        source_states,
        actual_concept_lengths,
    )
    torch.testing.assert_close(actual_sources, expected_sources)
    assert actual_source_lengths == expected_source_lengths

    expected_decoder_sources = repeat_shift_reference(expected_sources, expected_source_lengths, False)
    actual_decoder_sources = model._repeat_shift(
        actual_sources,
        actual_source_lengths,
        token_lengths,
        prepend_zero=False,
    )
    torch.testing.assert_close(actual_decoder_sources, expected_decoder_sources)

    context = model._build_concept_context(
        actual_merged,
        actual_concept_lengths,
        config.concept_chunk_size,
    )
    expected_positions = torch.cat(
        [torch.arange(length) * config.concept_chunk_size for length in expected_concept_lengths]
    ).unsqueeze(0)
    torch.testing.assert_close(context.position_ids, expected_positions)
    torch.testing.assert_close(
        context.cu_seq_lens_q,
        torch.tensor([0, *torch.tensor(expected_concept_lengths).cumsum(0).tolist()], dtype=torch.int32),
    )

    if len(token_lengths) == 1:
        contiguous = torch.randn(1, 2, 3, generator=generator)
        assert model._cat_if_multiple([contiguous], dim=1) is contiguous


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

    def test_grouped_stage_compile_with_fsdp(self):
        self.create_pg("cuda")
        baseline_config = _stage_group_tiny_config(stage_group_size=1)
        baseline_config.attention.attn_impl = "flash_attention"
        baseline = baseline_config.build().cuda().bfloat16().eval()
        _initialize(baseline)

        grouped_config = _stage_group_tiny_config(stage_group_size=4)
        grouped_config.attention.attn_impl = "flash_attention"
        grouped_config.compile_cfg = None
        grouped = grouped_config.build().cuda().bfloat16().eval()

        baseline_parameters = list(baseline.named_parameters())
        grouped_parameters = list(grouped.named_parameters())
        with torch.no_grad():
            for (baseline_name, baseline_parameter), (grouped_name, grouped_parameter) in zip(
                baseline_parameters, grouped_parameters, strict=True
            ):
                assert grouped.to_hf_key_list(grouped_name) == baseline.to_hf_key_list(baseline_name)
                grouped_parameter.copy_(baseline_parameter)

        input_ids = torch.arange(1, 9, device="cuda").unsqueeze(0)
        baseline_logits = baseline(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
        generator = torch.Generator(device="cuda").manual_seed(31)
        output_grad = torch.randn(baseline_logits.shape, generator=generator, device="cuda")
        output_grad /= output_grad.numel()
        baseline_logits.float().backward(output_grad)

        grouped.fully_shard(
            FSDPConfig(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                recompute_ratio=0,
                reshard_after_forward=False,
            )
        )
        all_groups = (
            *grouped.encoder_stage_groups,
            *grouped.concept_stage_groups,
            *grouped.decoder_stage_groups,
        )
        hook_counts = {id(group): [0, 0] for group in all_groups}
        hook_compile_states: list[bool] = []

        def _record_pre_hook(module, _args):
            hook_counts[id(module)][0] += 1
            hook_compile_states.append(torch.compiler.is_compiling())

        def _record_post_hook(module, _args, _output):
            hook_counts[id(module)][1] += 1
            hook_compile_states.append(torch.compiler.is_compiling())

        for group in all_groups:
            assert isinstance(group, FSDPModule)
            assert all(not isinstance(stage, FSDPModule) for stage in group)
            group.register_forward_pre_hook(_record_pre_hook)
            group.register_forward_hook(_record_post_hook)

        grouped_logits = grouped(SequenceContext.from_input_ids((input_ids,), device="cuda")).logits
        torch.testing.assert_close(grouped_logits.float(), baseline_logits.float(), rtol=0.02, atol=0.02)
        grouped_logits.float().backward(output_grad)

        assert all(counts == [1, 1] for counts in hook_counts.values())
        assert not any(hook_compile_states)
        for group in all_groups:
            assert hasattr(type(group).forward, "_torchdynamo_orig_callable")

        grouped_parameters = list(grouped.named_parameters())
        for (_, baseline_parameter), (_, grouped_parameter) in zip(
            baseline_parameters, grouped_parameters, strict=True
        ):
            assert grouped_parameter.grad is not None
            assert torch.isfinite(grouped_parameter.grad.to_local()).all()
            if baseline_parameter.grad is not None:
                torch.testing.assert_close(
                    grouped_parameter.grad.to_local().float(),
                    baseline_parameter.grad.float(),
                    rtol=0.05,
                    atol=0.05,
                )

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
    assert len(loaded) == 631
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


@pytest.mark.skipif(MODEL_PATH is None, reason="CONCEPT_OLMO_STAGE2_V1_PATH is not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the full checkpoint")
def test_full_checkpoint_hf_alignment(monkeypatch):
    """Compare all logits, next-token loss, and the parameter grad norm with HF.

    Both implementations run the same tokenizer-produced prompts independently.
    The optimized alignment case below owns packed-varlen coverage; keeping the
    execution shapes identical here preserves this test as the strict HF oracle.
    """
    assert MODEL_PATH is not None
    monkeypatch.setenv("XTUNER_HF_IMPL", "true")
    monkeypatch.setenv("XTUNER_DETERMINISTIC", "true")
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "0")
    device = torch.device("cuda")
    documents_cpu, document_lengths, shifted_labels, loss_weight, expected_cu_seq_lens = (
        _build_alignment_prompt_batch(MODEL_PATH, device)
    )

    shifted_label_documents = shifted_labels.split(document_lengths, dim=1)
    loss_weight_documents = loss_weight.split(document_lengths, dim=1)

    # This remote HF model ignores attention_mask, so run each tokenizer output
    # independently. XTuner uses the same per-prompt execution shapes below.
    torch.cuda.reset_peak_memory_stats(device)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).train()
    hf_document_outputs = []
    for document in documents_cpu:
        document = document.to(device)
        output = hf_model(input_ids=document, labels=document, use_cache=False)
        assert output.loss is not None
        hf_document_outputs.append(output)
    hf_logits_tensor = torch.cat([output.logits for output in hf_document_outputs], dim=1)
    hf_logits = hf_logits_tensor.detach().float().cpu()
    total_valid_tokens = sum(document_length - 1 for document_length in document_lengths)
    hf_native_loss_tensor = torch.stack(
        [
            output.loss * (document_length - 1)
            for output, document_length in zip(hf_document_outputs, document_lengths, strict=True)
        ]
    ).sum() / total_valid_tokens
    hf_native_loss = hf_native_loss_tensor.detach().float().item()
    hf_shared_loss_tensor = torch.stack(
        [
            (
                F.cross_entropy(
                    output.logits.float().reshape(-1, output.logits.shape[-1]),
                    document_labels.flatten(),
                    reduction="none",
                    ignore_index=-100,
                )
                * document_loss_weight.flatten()
            ).sum()
            for output, document_labels, document_loss_weight in zip(
                hf_document_outputs,
                shifted_label_documents,
                loss_weight_documents,
                strict=True,
            )
        ]
    ).sum()
    hf_shared_loss = hf_shared_loss_tensor.detach().float().item()

    # Keep the graph once so we can distinguish the HF-native mean-CE
    # backward from the exact loss reduction used by XTuner.
    hf_native_loss_tensor.backward(retain_graph=True)
    hf_native_grad_norm_fp32, hf_native_grad_norm_fp64, hf_native_grad_numel = _global_grad_norms(hf_model)
    hf_model.zero_grad(set_to_none=True)
    hf_shared_loss_tensor.backward()
    hf_shared_grad_norm_fp32, hf_shared_grad_norm_fp64, hf_shared_grad_numel = _global_grad_norms(hf_model)
    hf_peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30

    del hf_native_loss_tensor, hf_shared_loss_tensor, hf_logits_tensor, hf_document_outputs, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config = ConceptLMV22VQConfig.from_hf(MODEL_PATH).model_copy(
        update={
            "compile_cfg": False,
            "float8_cfg": None,
            "lm_head_vocab_padding_multiple": 1,
            "stage_group_size": 1,
        }
    )
    with torch.device("meta"):
        model = config.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
    loaded, unloaded, missing = model.from_hf(MODEL_PATH)
    assert len(loaded) == 631
    assert not unloaded
    assert not missing
    model.train()

    # Match HF's GEMM shapes exactly. Packed-varlen execution deliberately lives
    # in test_full_checkpoint_hf_optimized_varlen_alignment, where differences
    # from fused/compiled kernels are measured with optimized-path tolerances.
    document_seq_contexts = tuple(
        SequenceContext.from_input_ids((document,), device="cpu").to(device)
        for document in documents_cpu
    )
    for seq_ctx, document_length in zip(document_seq_contexts, document_lengths, strict=True):
        _assert_alignment_sequence_context(seq_ctx, (document_length,), [0, document_length])
    loss_config = CELossConfig(mode="eager", loss_reduction="token")
    loss_contexts = []
    for document_labels in shifted_label_documents:
        loss_context = loss_config.build(data={"shifted_labels": document_labels}, sp_mesh=None)
        assert loss_context is not None
        loss_contexts.append(loss_context)
    loss_contexts = loss_config.loss_ctx_cls.build_batches(loss_contexts)
    torch.cuda.reset_peak_memory_stats(device)
    xtuner_document_logits = []
    xtuner_document_losses = []
    for seq_ctx, loss_context in zip(document_seq_contexts, loss_contexts, strict=True):
        output = model(seq_ctx, loss_ctx={"lm": loss_context})
        assert output.loss is not None
        assert output.logits is not None
        xtuner_document_logits.append(output.logits)
        xtuner_document_losses.append(output.loss)
    xtuner_logits_tensor = torch.cat(xtuner_document_logits, dim=0)
    xtuner_full_logits = xtuner_logits_tensor.reshape_as(hf_logits).detach().float().cpu()
    xtuner_loss_tensor = torch.stack(xtuner_document_losses).sum()
    xtuner_loss = xtuner_loss_tensor.detach().float().item()
    xtuner_loss_tensor.backward()
    xtuner_grad_norm_fp32, xtuner_grad_norm_fp64, xtuner_grad_numel = _global_grad_norms(model)
    xtuner_peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30

    logits_metrics = _tensor_error_metrics(xtuner_full_logits, hf_logits)
    document_logits_metrics = []
    start = 0
    for prompt, document_length in zip(ALIGNMENT_PROMPTS, document_lengths, strict=True):
        end = start + document_length
        document_logits_metrics.append(
            {
                "prompt": prompt,
                "length": document_length,
                **_tensor_error_metrics(xtuner_full_logits[:, start:end], hf_logits[:, start:end]),
            }
        )
        start = end
    native_loss_abs_error = abs(xtuner_loss - hf_native_loss)
    native_loss_rel_error = native_loss_abs_error / max(abs(hf_native_loss), 1e-12)
    shared_loss_abs_error = abs(xtuner_loss - hf_shared_loss)
    shared_loss_rel_error = shared_loss_abs_error / max(abs(hf_shared_loss), 1e-12)
    native_grad_norm_fp32_abs_error = abs(xtuner_grad_norm_fp32 - hf_native_grad_norm_fp32)
    native_grad_norm_fp32_rel_error = native_grad_norm_fp32_abs_error / max(abs(hf_native_grad_norm_fp32), 1e-12)
    shared_grad_norm_fp32_abs_error = abs(xtuner_grad_norm_fp32 - hf_shared_grad_norm_fp32)
    shared_grad_norm_fp32_rel_error = shared_grad_norm_fp32_abs_error / max(abs(hf_shared_grad_norm_fp32), 1e-12)
    native_grad_norm_fp64_abs_error = abs(xtuner_grad_norm_fp64 - hf_native_grad_norm_fp64)
    native_grad_norm_fp64_rel_error = native_grad_norm_fp64_abs_error / max(abs(hf_native_grad_norm_fp64), 1e-12)
    shared_grad_norm_fp64_abs_error = abs(xtuner_grad_norm_fp64 - hf_shared_grad_norm_fp64)
    shared_grad_norm_fp64_rel_error = shared_grad_norm_fp64_abs_error / max(abs(hf_shared_grad_norm_fp64), 1e-12)
    metrics = {
        "prompts": list(ALIGNMENT_PROMPTS),
        "input_ids": [document.tolist() for document in documents_cpu],
        "document_lengths": list(document_lengths),
        "concatenated_cu_seq_lens": expected_cu_seq_lens,
        "execution": {"hf": "per_prompt", "xtuner": "per_prompt"},
        "logits": {
            "shape": list(hf_logits.shape),
            **logits_metrics,
            "per_document": document_logits_metrics,
        },
        "loss": {
            "hf_native": hf_native_loss,
            "hf_shared_reduction": hf_shared_loss,
            "xtuner": xtuner_loss,
            "native_abs_error": native_loss_abs_error,
            "native_rel_error": native_loss_rel_error,
            "shared_abs_error": shared_loss_abs_error,
            "shared_rel_error": shared_loss_rel_error,
        },
        "grad_norm_fp32": {
            "hf_native": hf_native_grad_norm_fp32,
            "hf_shared_reduction": hf_shared_grad_norm_fp32,
            "xtuner": xtuner_grad_norm_fp32,
            "native_abs_error": native_grad_norm_fp32_abs_error,
            "native_rel_error": native_grad_norm_fp32_rel_error,
            "shared_abs_error": shared_grad_norm_fp32_abs_error,
            "shared_rel_error": shared_grad_norm_fp32_rel_error,
            "hf_grad_numel": hf_native_grad_numel,
            "xtuner_grad_numel": xtuner_grad_numel,
        },
        "grad_norm_fp64_diagnostic": {
            "hf_native": hf_native_grad_norm_fp64,
            "hf_shared_reduction": hf_shared_grad_norm_fp64,
            "xtuner": xtuner_grad_norm_fp64,
            "native_abs_error": native_grad_norm_fp64_abs_error,
            "native_rel_error": native_grad_norm_fp64_rel_error,
            "shared_abs_error": shared_grad_norm_fp64_abs_error,
            "shared_rel_error": shared_grad_norm_fp64_rel_error,
            "hf_grad_numel": hf_native_grad_numel,
            "xtuner_grad_numel": xtuner_grad_numel,
        },
        "peak_memory_gib": {
            "hf": hf_peak_memory_gib,
            "xtuner": xtuner_peak_memory_gib,
        },
    }
    print(f"CONCEPT_OLMO_HF_ALIGNMENT={json.dumps(metrics, ensure_ascii=False, sort_keys=True)}")

    assert hf_native_grad_numel == hf_shared_grad_numel == xtuner_grad_numel
    assert logits_metrics["max_abs_error"] == 0.0
    assert native_loss_abs_error <= 1e-6
    assert shared_loss_abs_error == 0.0
    assert native_grad_norm_fp32_rel_error <= 5e-3
    assert shared_grad_norm_fp32_rel_error <= 5e-3


@pytest.mark.skipif(MODEL_PATH is None, reason="CONCEPT_OLMO_STAGE2_V1_PATH is not set")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the full checkpoint")
def test_full_checkpoint_hf_optimized_varlen_alignment(monkeypatch):
    """Compare the optimized packed compute graph with HF on three prompts.

    The checkpoint's remote code does not implement Transformers' selectable
    ``attn_implementation`` interface. It calls PyTorch SDPA directly instead.
    Force its Flash SDPA backend, and pin XTuner to external flash-attn v2
    varlen rather than the benchmark's FA3 path. Both sides therefore exercise
    FlashAttention-2 semantics, though they are different kernel libraries and
    are compared with optimized-path tolerances rather than bitwise equality.

    HuggingFace ignores ``attention_mask`` in this remote model, so it cannot
    serve as a one-shot packed-sequence oracle. Each prompt is therefore run
    independently by HF, while XTuner receives the three documents as one
    packed varlen sequence with independent position and attention boundaries.
    """
    assert MODEL_PATH is not None
    monkeypatch.delenv("XTUNER_HF_IMPL", raising=False)
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "0")
    monkeypatch.setenv("XTUNER_USE_FA3", "0")
    device = torch.device("cuda")

    # The flash-attention implementation is resolved when XTuner's ops module
    # is imported, before a pytest monkeypatch can change XTUNER_USE_FA3. Patch
    # that already-resolved callable as well, so this case cannot accidentally
    # exercise FA3 inherited from the parent test process.
    import xtuner.v1.ops.attn_imp as attn_imp
    from xtuner.v1.ops.flash_attn.gpu import flash_attn_varlen_func_v2

    if flash_attn_varlen_func_v2 is None:
        pytest.skip("flash-attn v2 is required for optimized HF parity")
    monkeypatch.setattr(attn_imp, "flash_attn_varlen_func", flash_attn_varlen_func_v2)

    documents_cpu, document_lengths, shifted_labels, loss_weight, expected_cu_seq_lens = (
        _build_alignment_prompt_batch(MODEL_PATH, device)
    )

    checkpoint_config = ConceptLMV22VQConfig.from_hf(MODEL_PATH)
    assert max(document_lengths) <= checkpoint_config.window_size[0]

    # The remote HF model cannot consume packed document boundaries. Running
    # the three prompts separately is the unambiguous oracle: attention,
    # positions, concept chunking, and shifted labels all restart per prompt.
    torch.cuda.reset_peak_memory_stats(device)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).train()
    hf_document_outputs = []
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        for document in documents_cpu:
            hf_document_outputs.append(
                hf_model(
                    input_ids=document.to(device),
                    use_cache=False,
                )
            )
        hf_logits_tensor = torch.cat([output.logits for output in hf_document_outputs], dim=1)
        hf_loss_tensor = F.cross_entropy(
            hf_logits_tensor.float().reshape(-1, hf_logits_tensor.shape[-1]),
            shifted_labels.flatten(),
            reduction="none",
            ignore_index=-100,
        )
        hf_loss_tensor = (hf_loss_tensor * loss_weight.flatten()).sum()
        hf_logits = hf_logits_tensor.detach().float().cpu()
        hf_loss = hf_loss_tensor.detach().float().item()
        hf_loss_tensor.backward()
    hf_grad_norm_fp32, hf_grad_norm_fp64, hf_grad_numel = _global_grad_norms(hf_model)
    hf_peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30

    del hf_loss_tensor, hf_logits_tensor, hf_document_outputs, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config = checkpoint_config.model_copy(
        update={
            "compile_cfg": None,
            "float8_cfg": None,
            "lm_head_vocab_padding_multiple": 8,
            "stage_group_size": 4,
        }
    )
    config.attention.attn_impl = "flash_attention"
    with torch.device("meta"):
        model = config.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
    loaded, unloaded, missing = model.from_hf(MODEL_PATH)
    assert len(loaded) == 631
    assert not unloaded
    assert not missing
    model.train()

    packed_seq_ctx = SequenceContext.from_input_ids(documents_cpu, device="cpu").to(device)
    _assert_alignment_sequence_context(packed_seq_ctx, document_lengths, expected_cu_seq_lens)

    loss_config = CELossConfig(mode="eager", loss_reduction="token")
    loss_context = loss_config.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
    assert loss_context is not None
    loss_context = loss_config.loss_ctx_cls.build_batches([loss_context])[0]
    torch.cuda.reset_peak_memory_stats(device)
    output = model(packed_seq_ctx, loss_ctx={"lm": loss_context})
    assert output.loss is not None
    assert output.logits is not None
    xtuner_logits = output.logits.reshape_as(hf_logits).detach().float().cpu()
    xtuner_loss_tensor = output.loss
    xtuner_loss = xtuner_loss_tensor.detach().float().item()
    xtuner_loss_tensor.backward()
    xtuner_grad_norm_fp32, xtuner_grad_norm_fp64, xtuner_grad_numel = _global_grad_norms(model)
    xtuner_peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30

    logits_metrics = _tensor_error_metrics(xtuner_logits, hf_logits)
    document_logits_metrics = []
    start = 0
    for prompt, document_length in zip(ALIGNMENT_PROMPTS, document_lengths, strict=True):
        end = start + document_length
        document_logits_metrics.append(
            {
                "prompt": prompt,
                "length": document_length,
                **_tensor_error_metrics(xtuner_logits[:, start:end], hf_logits[:, start:end]),
            }
        )
        start = end
    loss_abs_error = abs(xtuner_loss - hf_loss)
    loss_rel_error = loss_abs_error / max(abs(hf_loss), 1e-12)
    grad_norm_fp32_abs_error = abs(xtuner_grad_norm_fp32 - hf_grad_norm_fp32)
    grad_norm_fp32_rel_error = grad_norm_fp32_abs_error / max(abs(hf_grad_norm_fp32), 1e-12)
    grad_norm_fp64_abs_error = abs(xtuner_grad_norm_fp64 - hf_grad_norm_fp64)
    grad_norm_fp64_rel_error = grad_norm_fp64_abs_error / max(abs(hf_grad_norm_fp64), 1e-12)
    metrics = {
        "attention": {
            "hf": "torch_sdpa_flash",
            "xtuner": "flash_attn_varlen_v2",
        },
        "document_lengths": list(document_lengths),
        "cu_seq_lens": expected_cu_seq_lens,
        "optimized_xtuner": {
            "compile": True,
            "stage_group_size": config.stage_group_size,
            "lm_head_vocab_padding_multiple": config.lm_head_vocab_padding_multiple,
            "float8": False,
            "fsdp": False,
        },
        "logits": {
            "shape": list(hf_logits.shape),
            **logits_metrics,
            "per_document": document_logits_metrics,
        },
        "loss": {
            "hf": hf_loss,
            "xtuner": xtuner_loss,
            "abs_error": loss_abs_error,
            "rel_error": loss_rel_error,
        },
        "grad_norm_fp32": {
            "hf": hf_grad_norm_fp32,
            "xtuner": xtuner_grad_norm_fp32,
            "abs_error": grad_norm_fp32_abs_error,
            "rel_error": grad_norm_fp32_rel_error,
        },
        "grad_norm_fp64_diagnostic": {
            "hf": hf_grad_norm_fp64,
            "xtuner": xtuner_grad_norm_fp64,
            "abs_error": grad_norm_fp64_abs_error,
            "rel_error": grad_norm_fp64_rel_error,
        },
        "grad_numel": {
            "hf": hf_grad_numel,
            "xtuner": xtuner_grad_numel,
        },
        "peak_memory_gib": {
            "hf": hf_peak_memory_gib,
            "xtuner": xtuner_peak_memory_gib,
        },
    }
    print(
        "CONCEPT_OLMO_HF_OPTIMIZED_VARLEN_ALIGNMENT="
        f"{json.dumps(metrics, ensure_ascii=False, sort_keys=True)}"
    )

    padded_lm_head_rows = model.lm_head.weight.shape[0] - config.vocab_size
    assert padded_lm_head_rows == 2
    assert model.lm_head.weight.grad is not None
    assert not torch.count_nonzero(model.lm_head.weight.grad[config.vocab_size :])
    assert xtuner_grad_numel == hf_grad_numel + padded_lm_head_rows * config.hidden_size

    # Measured full-checkpoint budgets for the optimized compute graph. Unlike
    # the strict case above, this path intentionally changes GEMM shapes,
    # reduction order, LM-head padding, and the FlashAttention implementation.
    assert logits_metrics["max_abs_error"] <= 1.0
    assert logits_metrics["mean_abs_error"] <= 0.1
    assert loss_rel_error <= 2e-3
    assert grad_norm_fp32_rel_error <= 0.08
    assert grad_norm_fp64_rel_error <= 0.08
