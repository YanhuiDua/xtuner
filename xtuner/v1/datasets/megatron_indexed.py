"""Megatron-LM indexed token dataset support."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import torch

from .data_item import DataItem


_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_DTYPES = {
    1: np.dtype("uint8"),
    2: np.dtype("int8"),
    3: np.dtype("<i2"),
    4: np.dtype("<i4"),
    5: np.dtype("<i8"),
    6: np.dtype("<f8"),
    7: np.dtype("<f4"),
    8: np.dtype("<u2"),
}


def _normalize_prefix(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix in (".bin", ".idx"):
        path = path.with_suffix("")
    return path


def resolve_megatron_prefixes(path: str | Path) -> list[Path]:
    """Resolve a prefix, prefix-list text file, or directory.

    Args:
        path (str | Path): Dataset prefix, directory, or text file containing prefixes.

    Returns:
        list[Path]: Validated Megatron dataset prefixes.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".txt":
        prefixes = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            prefixes.append(_normalize_prefix(line.split()[-1]))
    elif path.is_dir():
        prefixes = [_normalize_prefix(prefix) for prefix in sorted(path.rglob("*.idx"))]
    else:
        prefixes = [_normalize_prefix(path)]

    if not prefixes:
        raise FileNotFoundError(f"No Megatron indexed dataset prefixes found under {path}")

    missing = [
        str(prefix)
        for prefix in prefixes
        if not prefix.with_suffix(".idx").is_file() or not prefix.with_suffix(".bin").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing .bin/.idx pair for Megatron prefixes: {missing}")
    return prefixes


class MegatronIndexedDataset(torch.utils.data.Dataset[DataItem]):
    """Expose one Megatron IndexedDataset prefix as XTuner data items.

    Args:
        anno_path (str | Path): Dataset prefix, optionally ending in ``.bin`` or ``.idx``.
        sample_ratio (float): Sampling ratio. Only 1.0 is currently supported.
        name (str): Dataset name.
        sequence_length (int | None): Fixed block length. When set, the token file is read as one continuous stream.
    """

    def __init__(
        self,
        anno_path: str | Path,
        sample_ratio: float = 1.0,
        name: str = "default",
        sequence_length: int | None = None,
        **_kwargs,
    ) -> None:
        super().__init__()
        if sample_ratio != 1.0:
            raise NotImplementedError("MegatronIndexedDataset currently requires sample_ratio=1.0")

        self.prefix = _normalize_prefix(anno_path)
        self.path = str(self.prefix)
        self.name = name
        self.idx_path = self.prefix.with_suffix(".idx")
        self.bin_path = self.prefix.with_suffix(".bin")

        with self.idx_path.open("rb") as idx_file:
            magic = idx_file.read(9)
            version = struct.unpack("<Q", idx_file.read(8))[0]
            dtype_code = idx_file.read(1)[0]
            self.sequence_count = struct.unpack("<Q", idx_file.read(8))[0]
            self.document_count = struct.unpack("<Q", idx_file.read(8))[0]

        if magic != _INDEX_MAGIC:
            raise ValueError(f"Invalid Megatron index magic in {self.idx_path}: {magic!r}")
        if version != _INDEX_VERSION:
            raise ValueError(f"Unsupported Megatron index version {version} in {self.idx_path}")
        if dtype_code not in _DTYPES:
            raise ValueError(f"Unsupported Megatron dtype code {dtype_code} in {self.idx_path}")

        self.dtype = _DTYPES[dtype_code]
        if self.dtype.kind not in "iu":
            raise TypeError(f"Token IDs must use an integer dtype, got {self.dtype} in {self.bin_path}")

        sizes_offset = 34
        pointers_offset = sizes_offset + self.sequence_count * np.dtype("<i4").itemsize
        documents_offset = pointers_offset + self.sequence_count * np.dtype("<i8").itemsize
        expected_idx_size = documents_offset + self.document_count * np.dtype("<i8").itemsize
        actual_idx_size = self.idx_path.stat().st_size
        if actual_idx_size != expected_idx_size:
            raise ValueError(
                f"Invalid Megatron index size for {self.idx_path}: expected {expected_idx_size}, got {actual_idx_size}"
            )

        self._pointers_offset = pointers_offset
        self._documents_offset = documents_offset
        self._fd_pid: int | None = None
        self._idx_fd: int | None = None
        self._bin_fd: int | None = None
        self._block_length = sequence_length
        self._block_stride: int | None = None

        bin_size = self.bin_path.stat().st_size
        if bin_size % self.dtype.itemsize:
            raise ValueError(f"Invalid token file size {bin_size} for dtype {self.dtype} in {self.bin_path}")

        if sequence_length is not None:
            if sequence_length < 2:
                raise ValueError("sequence_length must be at least 2 to construct next-token samples")
            total_tokens = bin_size // self.dtype.itemsize
            self._block_stride = sequence_length - 1
            self._sample_count = max(0, 1 + (total_tokens - sequence_length) // self._block_stride)
            self._sizes = np.broadcast_to(np.asarray(sequence_length, dtype="<i4"), (self._sample_count,))
            self._pointers = None
            self._document_indices = None
            self._bin = None
            return

        try:
            self._sizes = np.memmap(
                self.idx_path, mode="r", dtype="<i4", offset=sizes_offset, shape=(self.sequence_count,)
            )
            self._pointers = np.memmap(
                self.idx_path, mode="r", dtype="<i8", offset=pointers_offset, shape=(self.sequence_count,)
            )
            self._document_indices = np.memmap(
                self.idx_path, mode="r", dtype="<i8", offset=documents_offset, shape=(self.document_count,)
            )
            self._bin = np.memmap(self.bin_path, mode="r", dtype=self.dtype)
        except OSError:
            self._sizes = np.fromfile(self.idx_path, dtype="<i4", count=self.sequence_count, offset=sizes_offset)
            self._pointers = None
            self._document_indices = None
            self._bin = None

        self._sample_count = self.sequence_count
        if self.sequence_count:
            final_byte = self._get_pointer(self.sequence_count - 1) + int(self._sizes[-1]) * self.dtype.itemsize
            if final_byte > bin_size:
                raise ValueError(
                    f"Megatron index points beyond {self.bin_path}: final byte {final_byte}, file size {bin_size}"
                )

    @property
    def num_tokens(self) -> np.ndarray:
        """Return per-sample token counts.

        Returns:
            np.ndarray: Per-sample token counts.
        """
        return self._sizes

    @property
    def proxy_attn_flops(self) -> np.ndarray:
        """Return a monotonic per-sample attention cost proxy.

        Returns:
            np.ndarray: Per-sample attention cost proxy.
        """
        return self._sizes

    @property
    def document_indices(self) -> np.ndarray:
        """Return Megatron document boundary indices.

        Returns:
            np.ndarray: Document boundary indices.
        """
        if self._document_indices is None:
            self._document_indices = np.fromfile(
                self.idx_path, dtype="<i8", count=self.document_count, offset=self._documents_offset
            )
        return self._document_indices

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, item: int) -> DataItem:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)

        size = int(self._sizes[item])
        if self._block_length is not None:
            assert self._block_stride is not None
            byte_pointer = item * self._block_stride * self.dtype.itemsize
        else:
            byte_pointer = self._get_pointer(item)
        if byte_pointer % self.dtype.itemsize:
            raise ValueError(f"Unaligned pointer {byte_pointer} for item {item} in {self.idx_path}")
        if self._bin is not None:
            token_pointer = byte_pointer // self.dtype.itemsize
            token_array = self._bin[token_pointer : token_pointer + size]
        else:
            self._ensure_file_descriptors()
            assert self._bin_fd is not None
            raw_tokens = os.pread(self._bin_fd, size * self.dtype.itemsize, byte_pointer)
            if len(raw_tokens) != size * self.dtype.itemsize:
                raise EOFError(f"Could not read {size} tokens for item {item} from {self.bin_path}")
            token_array = np.frombuffer(raw_tokens, dtype=self.dtype)
        input_ids = token_array.astype(np.int64, copy=False).tolist()
        labels = input_ids.copy()
        if labels:
            labels[0] = -100
        return DataItem(
            input_ids=input_ids,
            labels=labels,
            num_tokens=size,
            proxy_attn_flops=float(size),
        )

    def _ensure_file_descriptors(self) -> None:
        pid = os.getpid()
        if self._fd_pid == pid:
            return
        for file_descriptor in (self._idx_fd, self._bin_fd):
            if file_descriptor is not None:
                os.close(file_descriptor)
        self._idx_fd = os.open(self.idx_path, os.O_RDONLY)
        self._bin_fd = os.open(self.bin_path, os.O_RDONLY)
        self._fd_pid = pid

    def _get_pointer(self, item: int) -> int:
        if self._pointers is not None:
            return int(self._pointers[item])
        self._ensure_file_descriptors()
        assert self._idx_fd is not None
        raw_pointer = os.pread(self._idx_fd, 8, self._pointers_offset + item * 8)
        if len(raw_pointer) != 8:
            raise EOFError(f"Could not read pointer for item {item} from {self.idx_path}")
        return struct.unpack("<q", raw_pointer)[0]

    def __del__(self) -> None:
        for file_descriptor in (getattr(self, "_idx_fd", None), getattr(self, "_bin_fd", None)):
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError:
                    continue
