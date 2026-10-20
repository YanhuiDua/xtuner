"""Compare matching fixed-length text iteration timings without modifying experiment artifacts."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def read_xtuner(path: Path) -> dict[int, dict[str, Any]]:
    """Read complete JSONL records, tolerating only an incomplete final row.

    Args:
        path (Path): xTuner metrics JSONL path.

    Returns:
        dict[int, dict[str, Any]]: Records keyed by zero-based step.
    """
    text = path.read_text()
    lines = text.splitlines()
    rows: dict[int, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        step = row["step"]
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step in rows:
            raise ValueError(f"Invalid or duplicate xTuner step: {step}")
        rows[step] = row
    return rows


def read_megatron(path: Path) -> dict[int, dict[str, Any]]:
    """Read timing lines for one Megatron run and normalize iteration numbering.

    Args:
        path (Path): Rank-zero log with one-based iteration numbers.

    Returns:
        dict[int, dict[str, Any]]: Records keyed by zero-based step.
    """
    pattern = re.compile(
        r"\biteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.eE+-]+)"
    )
    optional = {
        "mfu_percent": r"MFU[^|]*?:\s*([\d.eE+-]+)%",
        "tflops_per_gpu": r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.eE+-]+)",
    }
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        step = int(match[1]) - 1
        if step < 0 or step in rows:
            raise ValueError(f"Invalid or duplicate Megatron iteration: {step + 1}")
        row: dict[str, Any] = {"seconds": float(match[2]) / 1000}
        for key, expression in optional.items():
            value = re.search(expression, line)
            if value is not None:
                row[key] = float(value[1])
        rows[step] = row
    return rows


def summarize(rows: list[dict[str, Any]], tokens: int, world: int) -> dict[str, Any]:
    """Aggregate elapsed time and processed-token throughput.

    Args:
        rows (list[dict[str, Any]]): Selected per-step records.
        tokens (int): Global full-sequence tokens per optimizer update.
        world (int): Number of GPUs.

    Returns:
        dict[str, Any]: Timing, throughput and available recorded MFU metrics.
    """
    seconds = [float(row["seconds"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("Iteration times must be finite and positive")
    total = sum(seconds)
    result: dict[str, Any] = {
        "steps": len(rows), "seconds_total": total,
        "seconds_mean": statistics.mean(seconds), "seconds_median": statistics.median(seconds),
        "seconds_min": min(seconds), "seconds_max": max(seconds),
        "aggregate_global_tps": len(rows) * tokens / total,
        "aggregate_tps_per_gpu": len(rows) * tokens / total / world,
    }
    for key in ("mfu_percent", "tflops_per_gpu"):
        if all(key in row for row in rows):
            values = [float(row[key]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite recorded {key}")
            result[key + "_mean"] = statistics.mean(values)
            result[key + "_median"] = statistics.median(values)
    if all("peak_memory_gib" in row for row in rows):
        result["rank0_per_iteration_peak_allocated_gib_max"] = max(float(row["peak_memory_gib"]) for row in rows)
    return result


def main() -> None:
    """Print a read-only summary for overlapping steps in two recorded runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtuner-metrics", required=True, type=Path)
    parser.add_argument("--megatron-log", required=True, type=Path)
    parser.add_argument("--global-batch-size", required=True, type=int)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--skip-first", type=int, default=10)
    parser.add_argument("--exclude-step", type=int, action="append", default=[])
    args = parser.parse_args()
    if min(args.global_batch_size, args.sequence_length, args.world_size) <= 0 or args.skip_first < 0:
        parser.error("GBS, sequence length and world size must be positive; skip-first must be nonnegative")
    xtuner, megatron = read_xtuner(args.xtuner_metrics), read_megatron(args.megatron_log)
    excluded = set(args.exclude_step)
    steps = sorted(s for s in xtuner.keys() & megatron.keys() if s >= args.skip_first and s not in excluded)
    if not steps:
        parser.error("No overlapping steps remain after startup/profiling exclusions")
    tokens = args.global_batch_size * args.sequence_length
    left = summarize([xtuner[s] for s in steps], tokens, args.world_size)
    right = summarize([megatron[s] for s in steps], tokens, args.world_size)
    print(json.dumps({
        "status": "TIMINGS_SUMMARIZED_NOT_A_PARITY_OR_COMPLETION_CHECK",
        "assumed_config": {"global_batch_size": args.global_batch_size, "sequence_length": args.sequence_length,
                           "world_size": args.world_size, "throughput_tokens": "processed_including_masked"},
        "selected_zero_based_steps": steps, "skip_first": args.skip_first, "excluded_steps": sorted(excluded),
        "xtuner_only_steps": sorted(xtuner.keys() - megatron.keys()),
        "megatron_only_steps": sorted(megatron.keys() - xtuner.keys()),
        "xtuner": left, "megatron": right,
        "elapsed_ratio_xtuner_over_megatron": left["seconds_total"] / right["seconds_total"],
        "timing_scope": "Timers may include different work; no kernel attribution or inferred xTuner MFU.",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
