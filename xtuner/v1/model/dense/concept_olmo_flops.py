# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class ConceptOlmoConfigLike(Protocol):
    max_position_embeddings: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    concept_encoder_layers: int
    concept_decoder_layers: int
    concept_special_layers: int
    concept_chunk_size: int
    concept_v22_vq_codebook_size: int
    concept_v22_vq_num_codebooks: int
    window_size: tuple[int, int]
    window_attn_skip_freq: int


class MFUConfig(BaseModel):
    """Configuration for model FLOPs utilization logging.

    By default, the dense BF16/FP16 peak throughput is inferred from the runtime
    accelerator name. ``peak_tflops_per_gpu`` can override the detected value
    for unsupported accelerators, quantized training, or a custom power/clock
    limit. Sparse peak numbers would make the reported MFU artificially low.
    """

    model_config = ConfigDict(extra="forbid")

    sequence_length: int = Field(gt=0)
    peak_tflops_per_gpu: float | None = Field(default=None, gt=0)


def get_peak_tflops_per_gpu(device_name: str) -> float:
    """Return dense BF16/FP16 Tensor Core peak throughput for one GPU.

    The lookup follows TorchTitan's runtime device-name based MFU accounting.
    Unknown devices fail explicitly instead of silently using another GPU's
    peak, because an incorrect denominator produces a plausible but invalid MFU.
    """

    normalized_name = device_name.casefold()
    if "a100" in normalized_name:
        return 312.0
    if "a6000" in normalized_name:
        return 154.85
    if "h100" in normalized_name:
        if "nvl" in normalized_name:
            return 835.0
        if "pcie" in normalized_name:
            return 756.0
        return 989.0
    if "h200" in normalized_name:
        return 989.0
    if "h20" in normalized_name:
        return 148.0
    # Check Grace Blackwell names before B200/B300 because they contain the
    # shorter model name as a substring.
    if "gb200" in normalized_name or "gb300" in normalized_name:
        return 2500.0
    if "b200" in normalized_name or "b300" in normalized_name:
        return 2250.0
    if "l40s" in normalized_name:
        return 362.0

    raise ValueError(
        f"BF16/FP16 peak TFLOPS is unknown for accelerator {device_name!r}; "
        "set MFUConfig.peak_tflops_per_gpu explicitly"
    )


@dataclass(frozen=True)
class ConceptOlmoFlopsBreakdown:
    """Forward and training FLOPs for one Concept OLMo input sequence."""

    sequence_length: int
    concept_sequence_length: int
    encoder_transformer_flops: int
    encoder_route_flops: int
    concept_transformer_flops: int
    concept_route_flops: int
    concept_prediction_flops: int
    decoder_transformer_flops: int
    decoder_route_flops: int
    lm_head_flops: int

    @property
    def forward_flops(self) -> int:
        return sum(
            (
                self.encoder_transformer_flops,
                self.encoder_route_flops,
                self.concept_transformer_flops,
                self.concept_route_flops,
                self.concept_prediction_flops,
                self.decoder_transformer_flops,
                self.decoder_route_flops,
                self.lm_head_flops,
            )
        )

    @property
    def training_flops(self) -> int:
        # For matrix multiplications, backward is twice the forward cost: one
        # multiplication for each input gradient. Activation recomputation is
        # deliberately excluded because MFU measures useful model FLOPs.
        return 3 * self.forward_flops


@dataclass(frozen=True)
class MFUMetrics:
    model_flops_per_step: int
    tflops_per_gpu: float
    mfu: float


class ConceptOlmoFlopsEstimator:
    """Analytical FLOPs estimator for ConceptLM V2.2 VQ models.

    The estimator counts GEMMs, causal/sliding attention matrix products, and
    the learned depth/residual routing reductions. A multiply-add is two FLOPs.
    Embedding lookups, normalization, activations, softmax, elementwise ops, and
    communication are omitted, matching the usual transformer MFU convention.

    The requested sequence length represents one unpacked document. Packed
    batches with multiple document boundaries do less attention work and need
    their individual lengths for an exact attention count.
    """

    def __init__(self, model_config: ConceptOlmoConfigLike):
        self.config = model_config
        if model_config.concept_chunk_size <= 0:
            raise ValueError("concept_chunk_size must be positive")
        if model_config.window_attn_skip_freq <= 0:
            raise ValueError("window_attn_skip_freq must be positive")

    @staticmethod
    def _causal_attention_pairs(sequence_length: int, left_window: int | None) -> int:
        if left_window is None or left_window < 0:
            return sequence_length * (sequence_length + 1) // 2

        keys_per_query = left_window + 1
        warmup_queries = min(sequence_length, keys_per_query)
        return (
            warmup_queries * (warmup_queries + 1) // 2
            + (sequence_length - warmup_queries) * keys_per_query
        )

    @classmethod
    def _attention_forward_flops(
        cls,
        sequence_length: int,
        hidden_size: int,
        left_window: int | None,
    ) -> int:
        # QK^T and attention-value each cost two FLOPs per attended hidden
        # element, hence four FLOPs per causal attention pair.
        return (
            4 * cls._causal_attention_pairs(sequence_length, left_window) * hidden_size
        )

    @staticmethod
    def _route_forward_flops(
        sequence_length: int, hidden_size: int, num_sources: int
    ) -> int:
        if num_sources <= 0:
            return 0
        # w1: h -> r, w2: r -> r, followed by the weighted reduction of r
        # source states. Biases and elementwise operations are not counted.
        return (
            2 * sequence_length * hidden_size * num_sources
            + 2 * sequence_length * num_sources**2
            + 2 * sequence_length * num_sources * hidden_size
        )

    def _transformer_stack_forward_flops(
        self, sequence_length: int, num_layers: int
    ) -> int:
        config = self.config
        hidden_size = config.hidden_size
        linear_parameters_per_layer = (
            4 * hidden_size**2 + 3 * hidden_size * config.intermediate_size
        )
        linear_flops_per_layer = 2 * sequence_length * linear_parameters_per_layer

        total = 0
        for layer_idx in range(num_layers):
            is_full_attention = (layer_idx + 1) % config.window_attn_skip_freq == 0
            left_window = None if is_full_attention else config.window_size[0]
            total += linear_flops_per_layer
            total += self._attention_forward_flops(
                sequence_length, hidden_size, left_window
            )
        return total

    def _depth_route_forward_flops(self, sequence_length: int, num_layers: int) -> int:
        return sum(
            self._route_forward_flops(
                sequence_length, self.config.hidden_size, layer_idx + 2
            )
            for layer_idx in range(num_layers)
        )

    def estimate(self, sequence_length: int) -> ConceptOlmoFlopsBreakdown:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence_length {sequence_length} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        config = self.config
        hidden_size = config.hidden_size
        concept_length = max(1, sequence_length // config.concept_chunk_size)

        encoder_transformer_flops = self._transformer_stack_forward_flops(
            sequence_length,
            config.concept_encoder_layers,
        )
        encoder_route_flops = self._depth_route_forward_flops(
            sequence_length,
            config.concept_encoder_layers,
        )

        concept_transformer_flops = self._transformer_stack_forward_flops(
            concept_length,
            config.concept_special_layers,
        )
        concept_route_flops = self._depth_route_forward_flops(
            concept_length,
            config.concept_special_layers,
        )
        concept_route_flops += (
            config.concept_special_layers
            * self._route_forward_flops(
                concept_length,
                hidden_size,
                config.concept_encoder_layers - 1,
            )
        )
        concept_prediction_flops = (
            2
            * concept_length
            * hidden_size
            * config.concept_v22_vq_codebook_size
            * (config.concept_v22_vq_num_codebooks + 1)
        )

        decoder_transformer_flops = self._transformer_stack_forward_flops(
            sequence_length,
            config.concept_decoder_layers,
        )
        decoder_route_flops = self._depth_route_forward_flops(
            sequence_length,
            config.concept_decoder_layers,
        )
        decoder_route_flops += config.concept_decoder_layers * (
            self._route_forward_flops(
                sequence_length,
                hidden_size,
                config.concept_encoder_layers,
            )
            + self._route_forward_flops(
                sequence_length,
                hidden_size,
                config.concept_special_layers,
            )
        )

        lm_head_flops = 2 * sequence_length * hidden_size * config.vocab_size
        return ConceptOlmoFlopsBreakdown(
            sequence_length=sequence_length,
            concept_sequence_length=concept_length,
            encoder_transformer_flops=encoder_transformer_flops,
            encoder_route_flops=encoder_route_flops,
            concept_transformer_flops=concept_transformer_flops,
            concept_route_flops=concept_route_flops,
            concept_prediction_flops=concept_prediction_flops,
            decoder_transformer_flops=decoder_transformer_flops,
            decoder_route_flops=decoder_route_flops,
            lm_head_flops=lm_head_flops,
        )


class ConceptOlmoMFUCalculator:
    """Convert Concept OLMo model FLOPs into per-step TFLOPS and MFU."""

    def __init__(
        self,
        model_config: ConceptOlmoConfigLike,
        mfu_config: MFUConfig,
        global_batch_size: int,
        world_size: int,
        device_name: str | None = None,
    ):
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        self.mfu_config = mfu_config
        self.global_batch_size = global_batch_size
        self.world_size = world_size
        self.device_name = device_name
        if mfu_config.peak_tflops_per_gpu is None:
            if device_name is None:
                raise ValueError(
                    "device_name is required when peak_tflops_per_gpu is not configured"
                )
            self.peak_tflops_per_gpu = get_peak_tflops_per_gpu(device_name)
            self.peak_tflops_source = "auto"
        else:
            self.peak_tflops_per_gpu = mfu_config.peak_tflops_per_gpu
            self.peak_tflops_source = "configured"
        self.breakdown = ConceptOlmoFlopsEstimator(model_config).estimate(
            mfu_config.sequence_length
        )

    @property
    def model_flops_per_step(self) -> int:
        return self.breakdown.training_flops * self.global_batch_size

    def compute(self, step_time: float) -> MFUMetrics:
        if step_time <= 0:
            raise ValueError("step_time must be positive")

        tflops_per_gpu = self.model_flops_per_step / step_time / self.world_size / 1e12
        return MFUMetrics(
            model_flops_per_step=self.model_flops_per_step,
            tflops_per_gpu=tflops_per_gpu,
            mfu=tflops_per_gpu / self.peak_tflops_per_gpu,
        )
