# Unit test path selection

GitHub Actions [`unit_test`](../../.github/workflows/unit_test.yaml) maps changed paths to pytest directories via [`select_unittest_targets.py`](select_unittest_targets.py). One cluster job still runs a single pytest command.

## Triggers

| Event | What runs |
| --- | --- |
| Pull request to `master` | Path-selected tests from the PR diff |
| Nightly schedule (23:00 Beijing) | Full non-RL suite: `pytest tests --ignore=tests/rl` |
| Manual `workflow_dispatch` | Checkout `ref`. Empty `changed_files` → full non-RL suite. Otherwise treat the list as a simulated diff (newline or comma separated; directory prefixes and `*` / `**` globs are OK). |

Concurrency: a new PR push cancels the previous run of that PR; overlapping nightlies cancel each other; each manual run is isolated.

Matching is **directory prefix**, not a single file. These all select `tests/datasets`:

```text
tests/datasets
tests/datasets/
tests/datasets/**
tests/datasets/test_dataloader.py
tests/datasets/nested/foo.py
```

The same prefix rule applies under `xtuner/v1/<package>/`.

## Test-tree changes (`tests/<pkg>/`)

A change only under a test package runs that package. It does **not** apply source-side dependency expansion.

| Changed path | Pytest |
| --- | --- |
| `tests/chat_template/` | `pytest tests/chat_template` |
| `tests/datasets/` | `pytest tests/datasets` |
| `tests/engine/` | `pytest tests/engine` |
| `tests/float8/` | `pytest tests/float8` |
| `tests/loss/` | `pytest tests/loss` |
| `tests/model/` | `pytest tests/model` |
| `tests/module/` | `pytest tests/module` |
| `tests/ops/` | `pytest tests/ops` |
| `tests/optim/` | `pytest tests/optim` |
| `tests/patch/` | `pytest tests/patch` |
| `tests/profiler/` | `pytest tests/profiler` |
| `tests/rl/` | `pytest tests/rl` |
| `tests/train/` | `pytest tests/train` |
| `tests/utils/` | `pytest tests/utils` |
| `tests/writer/` | `pytest tests/writer` |
| `tests/resource/` | `pytest tests/datasets tests/model` |

## Source-tree changes (`xtuner/v1/<pkg>/`)

| Changed path | Pytest |
| --- | --- |
| `xtuner/v1/datasets/` | `pytest tests/datasets tests/train` |
| `xtuner/v1/model/` | `pytest tests/engine tests/model tests/module tests/train` |
| `xtuner/v1/module/` | `pytest tests/engine tests/module tests/train` |
| `xtuner/v1/ops/` | `pytest tests/engine tests/ops tests/train` |
| `xtuner/v1/loss/` | `pytest tests/engine tests/loss tests/train` |
| `xtuner/v1/optim/` | `pytest tests/engine tests/optim tests/train` |
| `xtuner/v1/engine/` | `pytest tests/engine tests/train` |
| `xtuner/v1/float8/` | `pytest tests/float8` |
| `xtuner/v1/patch/` | `pytest tests/patch` |
| `xtuner/v1/profiler/` | `pytest tests/profiler` |
| `xtuner/v1/train/` | `pytest tests/train` |
| `xtuner/v1/utils/` | `pytest tests/utils` |
| `xtuner/v1/rl/` | `pytest tests/rl` |
| `xtuner/v1/_writer/` | `pytest tests/writer` |
| `xtuner/v1/data_proto/` | `pytest tests/chat_template` |

RL and SFT do not pull each other in. `tests/rl` is included only when the diff (or simulated files) touches `xtuner/v1/rl/` or `tests/rl/`.

## Full-suite fallback

Any of these forces the full command (plus `tests/rl` if the same diff also touches RL):

- `ci/`
- `tests/conftest.py`
- `xtuner/v1/config/`
- `xtuner/v1/__init__.py`
- `.github/workflows/unit_test.yaml`
- unknown `xtuner/` or `tests/` paths
- empty diff / selector failure

Default full command: `pytest tests --ignore=tests/rl`.

Rules are covered by [`test_select_unittest_targets.py`](test_select_unittest_targets.py).
