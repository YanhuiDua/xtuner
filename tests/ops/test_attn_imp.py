from types import SimpleNamespace

import torch

from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.ops.attn_imp import _create_windowed_grouped_causal_mask


def test_rope_parameters_support_rotary_base():
    hf_config = SimpleNamespace(
        rope_parameters=None,
        rope_scaling=None,
        rope_theta=None,
        rotary_base=500000,
    )

    config = RopeParametersConfig.from_hf_config(hf_config)

    assert config is not None
    assert config.rope_theta == 500000


def test_eager_window_boundary_is_inclusive():
    document_ids = torch.zeros((1, 6), dtype=torch.long)

    mask = _create_windowed_grouped_causal_mask(document_ids, window_size=4)

    assert mask[0, 5, 1] == 0
    assert torch.isneginf(mask[0, 5, 0])
