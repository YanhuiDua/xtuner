import struct
from pathlib import Path

import numpy as np

from xtuner.v1.datasets.megatron_indexed import MegatronIndexedDataset, resolve_megatron_prefixes


def _write_indexed_dataset(prefix: Path) -> None:
    sequences = [np.asarray([1, 2, 3], dtype="<i4"), np.asarray([4, 5], dtype="<i4")]
    with prefix.with_suffix(".bin").open("wb") as bin_file:
        for sequence in sequences:
            bin_file.write(sequence.tobytes())

    sizes = np.asarray([3, 2], dtype="<i4")
    pointers = np.asarray([0, 12], dtype="<i8")
    documents = np.asarray([0, 2], dtype="<i8")
    with prefix.with_suffix(".idx").open("wb") as idx_file:
        idx_file.write(b"MMIDIDX\x00\x00")
        idx_file.write(struct.pack("<Q", 1))
        idx_file.write(bytes([4]))
        idx_file.write(struct.pack("<Q", len(sizes)))
        idx_file.write(struct.pack("<Q", len(documents)))
        idx_file.write(sizes.tobytes())
        idx_file.write(pointers.tobytes())
        idx_file.write(documents.tobytes())


class TestMegatronIndexedDataset:
    def test_reads_indexed_sequences(self, tmp_path: Path) -> None:
        prefix = tmp_path / "tokens"
        _write_indexed_dataset(prefix)

        dataset = MegatronIndexedDataset(prefix)

        assert len(dataset) == 2
        assert dataset[0]["input_ids"] == [1, 2, 3]
        assert dataset[0]["labels"] == [-100, 2, 3]
        assert dataset[1]["input_ids"] == [4, 5]
        assert dataset.document_indices.tolist() == [0, 2]

    def test_reads_overlapping_fixed_blocks(self, tmp_path: Path) -> None:
        prefix = tmp_path / "tokens"
        _write_indexed_dataset(prefix)

        dataset = MegatronIndexedDataset(prefix, sequence_length=3)

        assert len(dataset) == 2
        assert dataset[0]["input_ids"] == [1, 2, 3]
        assert dataset[1]["input_ids"] == [3, 4, 5]

    def test_resolves_weighted_prefix_list(self, tmp_path: Path) -> None:
        prefix = tmp_path / "tokens"
        _write_indexed_dataset(prefix)
        prefix_list = tmp_path / "datasets.txt"
        prefix_list.write_text(f"0.5 {prefix}\n", encoding="utf-8")

        assert resolve_megatron_prefixes(prefix_list) == [prefix]
