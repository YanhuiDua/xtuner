# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KimiK3ChatTemplate(BaseModel):
    """Options for Kimi-K3's tokenizer-native XTML chat template.

    Kimi-K3 does not expose a Jinja template. Its remote tokenizer renders
    messages segment by segment so that structural XTML markers are special
    tokens while identical text supplied by a user remains ordinary text.
    Consequently this object stores rendering options only; rendering is
    delegated to ``tokenizer.apply_chat_template``.
    """

    model_config = ConfigDict(extra="forbid")

    thinking: bool = True
    thinking_effort: Literal["low", "high", "max"] | None = "max"
    stop_words: list[str] = Field(default_factory=lambda: ["<|end_of_msg|>", "[EOS]"])
