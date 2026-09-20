#!/usr/bin/env python3
"""Map changed files to a pytest command for GitHub Actions unit tests."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

FULL_NON_RL = "pytest tests --ignore=tests/rl"
FULL_WITH_RL = "pytest tests"

SRC_PACKAGE_TO_TEST = {
    "datasets": "datasets",
    "engine": "engine",
    "float8": "float8",
    "loss": "loss",
    "model": "model",
    "module": "module",
    "ops": "ops",
    "optim": "optim",
    "patch": "patch",
    "profiler": "profiler",
    "rl": "rl",
    "train": "train",
    "utils": "utils",
    "_writer": "writer",
    "data_proto": "chat_template",
}

TEST_PACKAGES = {
    "chat_template",
    "datasets",
    "engine",
    "float8",
    "loss",
    "model",
    "module",
    "ops",
    "optim",
    "patch",
    "profiler",
    "rl",
    "train",
    "utils",
    "writer",
}

DEPENDENCY_EXPAND = {
    "datasets": ("train",),
    "model": ("module", "engine", "train"),
    "module": ("engine", "train"),
    "ops": ("engine", "train"),
    "loss": ("engine", "train"),
    "optim": ("engine", "train"),
    "engine": ("train",),
}


def _normalize(path: str) -> str:
    path = path.strip().replace("\\", "/").strip("/")
    parts = [part for part in path.split("/") if part and part not in {"*", "**"}]
    return "/".join(parts)


def _path_in_dir(path: str, directory: str) -> bool:
    directory = directory.strip("/")
    return path == directory or path.startswith(directory + "/")


def _is_full_fallback_path(path: str) -> bool:
    if path in {"xtuner/v1/__init__.py", "tests/conftest.py", ".github/workflows/unit_test.yaml"}:
        return True
    if _path_in_dir(path, "xtuner/v1/config"):
        return True
    if _path_in_dir(path, "ci"):
        return True
    return False


def _source_package(path: str) -> str | None:
    if not _path_in_dir(path, "xtuner/v1"):
        return None
    rest = path[len("xtuner/v1") :].lstrip("/")
    if not rest or rest == "__init__.py":
        return None
    pkg = rest.split("/", 1)[0]
    if pkg.endswith(".py"):
        return None
    return pkg


def _test_package(path: str) -> str | None:
    if not _path_in_dir(path, "tests"):
        return None
    rest = path[len("tests") :].lstrip("/")
    if not rest:
        return None
    pkg = rest.split("/", 1)[0]
    if pkg in TEST_PACKAGES:
        return pkg
    if pkg == "resource":
        return "resource"
    return None


def _expand(packages: set[str]) -> set[str]:
    expanded = set(packages)
    for pkg in list(packages):
        expanded.update(DEPENDENCY_EXPAND.get(pkg, ()))
    return expanded


def select_test_targets(changed_files: Iterable[str]) -> tuple[set[str], bool]:
    """Return (test packages, include_rl).

    An empty package set means run the full suite. ``include_rl`` decides
    whether that full suite (or a targeted run) includes ``tests/rl``.
    """
    files = [_normalize(path) for path in changed_files if _normalize(path)]
    if not files:
        return set(), False

    source_pkgs: set[str] = set()
    test_pkgs: set[str] = set()
    fallback = False
    include_rl = False

    for path in files:
        if _path_in_dir(path, "xtuner/v1/rl") or _path_in_dir(path, "tests/rl"):
            include_rl = True
        if _is_full_fallback_path(path):
            fallback = True
            continue

        src_pkg = _source_package(path)
        if src_pkg is not None:
            mapped = SRC_PACKAGE_TO_TEST.get(src_pkg)
            if mapped is None:
                fallback = True
            else:
                source_pkgs.add(mapped)
            continue

        test_pkg = _test_package(path)
        if test_pkg == "resource":
            test_pkgs.update({"datasets", "model"})
            continue
        if test_pkg is not None:
            test_pkgs.add(test_pkg)
            continue

        fallback = True

    if fallback:
        return set(), include_rl

    targets = set(test_pkgs)
    targets.update(_expand(source_pkgs))
    if not targets:
        return set(), include_rl
    return targets, include_rl or "rl" in targets


def format_pytest_command(targets: set[str], include_rl: bool) -> str:
    if not targets:
        return FULL_WITH_RL if include_rl else FULL_NON_RL
    ordered = [f"tests/{name}" for name in sorted(targets)]
    return "pytest " + " ".join(ordered)


def select_pytest_command(changed_files: Iterable[str]) -> str:
    targets, include_rl = select_test_targets(changed_files)
    return format_pytest_command(targets, include_rl)


def is_only_rl(changed_files: Iterable[str]) -> bool:
    files = [_normalize(path) for path in changed_files if _normalize(path)]
    if not files:
        return False
    return all(
        _path_in_dir(path, "xtuner/v1/rl") or _path_in_dir(path, "tests/rl") for path in files
    )


def _write_github_output(path: str, pytest_command: str, only_rl: bool) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"pytest_command={pytest_command}\n")
        handle.write(f"only_rl={'true' if only_rl else 'false'}\n")


def _read_changed_files(args: argparse.Namespace) -> list[str]:
    if args.files:
        return list(args.files)
    return [line for line in sys.stdin.read().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select pytest targets from changed files")
    parser.add_argument("files", nargs="*", help="Changed file paths")
    parser.add_argument(
        "--github-output",
        default="",
        help="Append pytest_command and only_rl to a GitHub Actions output file",
    )
    args = parser.parse_args()
    files = _read_changed_files(args)
    command = select_pytest_command(files)
    print(command)
    if args.github_output:
        _write_github_output(args.github_output, command, is_only_rl(files))


if __name__ == "__main__":
    main()
