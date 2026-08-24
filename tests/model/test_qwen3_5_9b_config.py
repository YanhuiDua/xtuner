import torch

from xtuner.v1.model import Qwen3_5_VLTextDense9BConfig
from xtuner.v1.module.mtp import MTPConfig


class TestQwen35Dense9BConfig:
    def test_architecture(self) -> None:
        config = Qwen3_5_VLTextDense9BConfig()

        assert config.hidden_size == 4096
        assert config.intermediate_size == 12288
        assert config.num_hidden_layers == 32
        assert config.tie_word_embeddings is False
        assert config.layers_type.count("full_attention") == 8
        assert config.layers_type.count("linear_attention") == 24

    def test_accepts_dense_mtp(self) -> None:
        config = Qwen3_5_VLTextDense9BConfig(mtp_config=MTPConfig(num_layers=1))

        assert config.mtp_config is not None
        assert config.mtp_config.num_layers == 1

    def test_builds_dense_mtp_on_meta(self) -> None:
        config = Qwen3_5_VLTextDense9BConfig(mtp_config=MTPConfig(num_layers=1), compile_cfg=False)

        with torch.device("meta"):
            model = config.build()

        assert model.mtp_block is not None
        assert len(model.mtp_block.layers) == 1
        assert sum(parameter.numel() for parameter in model.parameters()) == 9_197_093_888
