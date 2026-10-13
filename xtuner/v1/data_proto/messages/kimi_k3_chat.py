# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.templates.kimi_k3 import KimiK3ChatTemplate
from xtuner.v1.utils import IGNORE_INDEX


def _input_ids(value: Any) -> list[int]:
    """Normalize the return types used by different Transformers versions."""
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _message_for_tokenizer(message: dict[str, Any]) -> dict[str, Any]:
    rendered = {key: value for key, value in message.items() if key not in {"loss", "thinking"}}
    if rendered.get("reasoning_content") is None and message.get("thinking") is not None:
        rendered["reasoning_content"] = message["thinking"]
    return {key: value for key, value in rendered.items() if value is not None}


class KimiK3ChatMessages(BaseModel):
    """Tokenize Kimi-K3 XTML conversations and construct SFT labels.

    The tokenizer's renderer is the source of truth. Assistant messages are rendered independently and aligned back
    into the fully rendered token stream. This avoids flattening XTML into a string, which is important because Kimi
    encodes structural markers and user-provided text with different special-token policies.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None

    def _apply_chat_template(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: list[dict[str, Any]],
        template: KimiK3ChatTemplate,
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
        thinking_effort: str | None,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tools": tools,
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "thinking": template.thinking,
            # Pass None explicitly: Kimi's tokenizer otherwise defaults a
            # missing thinking_effort to "max" and prepends a system option.
            "thinking_effort": thinking_effort,
        }
        return _input_ids(tokenizer.apply_chat_template(messages, **kwargs))

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: KimiK3ChatTemplate,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> dict:
        del kwargs
        raw_messages = self.messages
        messages = [_message_for_tokenizer(message) for message in raw_messages]
        input_ids = self._apply_chat_template(
            tokenizer,
            messages,
            chat_template,
            add_generation_prompt=add_generation_prompt,
            tools=self.tools,
            thinking_effort=chat_template.thinking_effort,
        )
        labels = [IGNORE_INDEX] * len(input_ids)
        generation_prefix = self._apply_chat_template(
            tokenizer,
            [],
            chat_template,
            add_generation_prompt=True,
            tools=None,
            thinking_effort=None,
        )

        # Structural XTML markers in user content are encoded as ordinary BPE
        # text by the Kimi tokenizer, so an independently rendered assistant
        # message (whose markers are special tokens) cannot match injected
        # user text. Advancing the cursor also disambiguates repeated replies.
        cursor = 0
        for index, (raw_message, message) in enumerate(zip(raw_messages, messages)):
            if message.get("role") != "assistant":
                continue
            assistant_ids = self._apply_chat_template(
                tokenizer,
                [message],
                chat_template,
                add_generation_prompt=False,
                tools=None,
                thinking_effort=None,
            )
            start = _find_subsequence(input_ids, assistant_ids, cursor)
            if start < 0:
                raise ValueError(f"Could not align Kimi-K3 assistant message at index {index}.")
            end = start + len(assistant_ids)
            if assistant_ids[: len(generation_prefix)] != generation_prefix:
                raise ValueError(
                    f"Kimi-K3 assistant message at index {index} does not start "
                    "with the tokenizer's generation prompt."
                )
            if raw_message.get("loss", True):
                target_start = start + len(generation_prefix)
                labels[target_start:end] = assistant_ids[len(generation_prefix) :]
            cursor = end

        return {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": len(input_ids),
        }


def _find_subsequence(sequence: list[int], subsequence: list[int], start: int) -> int:
    if not subsequence:
        return start
    limit = len(sequence) - len(subsequence) + 1
    for index in range(start, limit):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    return -1
