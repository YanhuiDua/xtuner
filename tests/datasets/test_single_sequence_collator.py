import pytest

from xtuner.v1.datasets.collator import single_sequence_sft_llm_collator
from xtuner.v1.datasets.config import DataloaderConfig


def test_single_sequence_collator_builds_one_cu_seq_range():
    instances = [
        [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4], "num_tokens": 4},
            {"input_ids": [5, 6, 7, 8, 9], "labels": [5, 6, 7, 8, 9], "num_tokens": 5},
        ]
    ]

    batch = single_sequence_sft_llm_collator(
        instances,  # type: ignore[arg-type]
        pack_max_length=9,
        padding_token_idx=0,
    )

    assert len(batch) == 1
    seq_ctx = batch[0]["seq_ctx"]
    assert seq_ctx.input_ids is not None
    assert seq_ctx.input_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert batch[0]["shifted_labels"].tolist() == [[2, 3, 4, 5, 6, 7, 8, 9]]
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, 8]
    assert seq_ctx.cu_seq_lens_k.tolist() == [0, 8]
    assert seq_ctx.position_ids is not None
    assert seq_ctx.position_ids.tolist() == [list(range(8))]
    assert seq_ctx.max_length_q.item() == 8
    assert seq_ctx.max_length_k.item() == 8
    assert seq_ctx.num_padding == 0


def test_single_sequence_collator_requires_exact_hard_pack():
    instances = [[{"input_ids": [1, 2, 3], "labels": [1, 2, 3], "num_tokens": 3}]]

    with pytest.raises(ValueError, match="requires exact-length packs"):
        single_sequence_sft_llm_collator(
            instances,  # type: ignore[arg-type]
            pack_max_length=9,
            padding_token_idx=0,
        )


def test_single_sequence_collator_accepts_unpacked_dict():
    source_tokens = list(range(9))
    batch = single_sequence_sft_llm_collator(
        [
            {
                "input_ids": source_tokens,
                "labels": source_tokens,
                "num_tokens": len(source_tokens),
            }
        ],  # type: ignore[arg-type]
        pack_max_length=9,
        padding_token_idx=0,
    )

    seq_ctx = batch[0]["seq_ctx"]
    assert seq_ctx.input_ids is not None
    assert seq_ctx.input_ids.shape == (1, 8)
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, 8]
    assert seq_ctx.num_padding == 0


def test_single_sequence_collator_produces_exact_8192_sequence():
    source_tokens = list(range(8193))
    instances = [
        [
            {
                "input_ids": source_tokens,
                "labels": source_tokens,
                "num_tokens": len(source_tokens),
            }
        ]
    ]

    batch = single_sequence_sft_llm_collator(
        instances,  # type: ignore[arg-type]
        pack_max_length=8193,
        padding_token_idx=0,
    )

    seq_ctx = batch[0]["seq_ctx"]
    assert seq_ctx.input_ids is not None
    assert seq_ctx.input_ids.shape == (1, 8192)
    assert batch[0]["shifted_labels"].shape == (1, 8192)
    assert seq_ctx.cu_seq_lens_q.tolist() == [0, 8192]
    assert seq_ctx.cu_seq_lens_k.tolist() == [0, 8192]
    assert seq_ctx.num_padding == 0


def test_dataloader_config_resolves_single_sequence_collator():
    config = DataloaderConfig(
        collator="single_sequence_sft_llm_collator",
        pack_level="hard",
        pack_max_length=8193,
    )

    assert config.build_collator() is single_sequence_sft_llm_collator
