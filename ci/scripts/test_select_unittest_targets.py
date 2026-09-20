import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_unittest_targets import (
    DEPENDENCY_EXPAND,
    SRC_PACKAGE_TO_TEST,
    TEST_PACKAGES,
    select_pytest_command,
)

PREFIX_SAMPLES = (
    "{dir}",
    "{dir}/",
    "{dir}/**",
    "{dir}/*",
    "{dir}/nested/foo.py",
)


def _pytest_for_packages(*names: str) -> str:
    return "pytest " + " ".join(f"tests/{name}" for name in sorted(names))


def _prefix_paths(directory: str) -> tuple[str, ...]:
    return tuple(sample.format(dir=directory) for sample in PREFIX_SAMPLES)


class TestSelectUnittestTargets(unittest.TestCase):
    def test_empty_falls_back_to_full_non_rl(self):
        self.assertEqual(select_pytest_command([]), "pytest tests --ignore=tests/rl")

    def test_all_test_packages_use_directory_prefix(self):
        for pkg in sorted(TEST_PACKAGES):
            expected = _pytest_for_packages(pkg)
            for path in _prefix_paths(f"tests/{pkg}"):
                self.assertEqual(select_pytest_command([path]), expected, msg=path)

    def test_all_source_packages_use_directory_prefix(self):
        for src_pkg, test_pkg in SRC_PACKAGE_TO_TEST.items():
            expected = _pytest_for_packages(test_pkg, *DEPENDENCY_EXPAND.get(test_pkg, ()))
            for path in _prefix_paths(f"xtuner/v1/{src_pkg}"):
                self.assertEqual(select_pytest_command([path]), expected, msg=path)

    def test_ci_falls_back_to_full_non_rl(self):
        self.assertEqual(
            select_pytest_command(["ci/scripts/CI_ENV.sh"]),
            "pytest tests --ignore=tests/rl",
        )
        self.assertEqual(
            select_pytest_command(["ci/**"]),
            "pytest tests --ignore=tests/rl",
        )

    def test_ci_and_rl_falls_back_to_full_with_rl(self):
        self.assertEqual(
            select_pytest_command(["ci/scripts/CI_ENV.sh", "xtuner/v1/rl/**"]),
            "pytest tests",
        )

    def test_conftest_falls_back(self):
        self.assertEqual(
            select_pytest_command(["tests/conftest.py"]),
            "pytest tests --ignore=tests/rl",
        )

    def test_resource_maps_to_datasets_and_model(self):
        expected = _pytest_for_packages("datasets", "model")
        for path in _prefix_paths("tests/resource"):
            self.assertEqual(select_pytest_command([path]), expected, msg=path)

    def test_unknown_xtuner_path_falls_back(self):
        self.assertEqual(
            select_pytest_command(["xtuner/tools/model_converters/split.py"]),
            "pytest tests --ignore=tests/rl",
        )

    def test_union_of_test_and_source_dirs(self):
        self.assertEqual(
            select_pytest_command(["tests/datasets/**", "xtuner/v1/model/**"]),
            "pytest tests/datasets tests/engine tests/model tests/module tests/train",
        )

    def test_only_rl_ignores_glob_suffix(self):
        from select_unittest_targets import is_only_rl

        self.assertTrue(is_only_rl(["tests/rl/**"]))
        self.assertTrue(is_only_rl(["xtuner/v1/rl"]))
        self.assertFalse(is_only_rl(["tests/rl/**", "xtuner/v1/model/**"]))


if __name__ == "__main__":
    unittest.main()
