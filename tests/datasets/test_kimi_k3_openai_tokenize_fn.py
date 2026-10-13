# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.utils import IGNORE_INDEX


class FakeKimiK3Tokenizer:
    """Small tokenizer-native XTML renderer used to test mask boundaries."""

    @staticmethod
    def _render(messages, tools, thinking, thinking_effort, add_generation_prompt):
        text = ""
        if tools:
            text += f'<|open|>message role="system" type="tool-declare"<|sep|>{json.dumps(tools)}'
            text += "<|close|>message<|sep|><|end_of_msg|>"
        if thinking and thinking_effort:
            text += f'<|open|>message role="system" type="thinking-effort"<|sep|>{thinking_effort}'
            text += "<|close|>message<|sep|><|end_of_msg|>"

        for message in messages:
            role = message["role"]
            text += f'<|open|>message role="{role}"<|sep|>'
            if role == "assistant":
                if thinking:
                    reasoning = message.get("reasoning_content", "")
                    text += f"<|open|>think<|sep|>{reasoning}<|close|>think<|sep|>"
                text += f"<|open|>response<|sep|>{message.get('content', '')}<|close|>response<|sep|>"
            else:
                text += message.get("content", "")
            text += "<|close|>message<|sep|><|end_of_msg|>"

        if add_generation_prompt:
            channel = "think" if thinking else "response"
            text += f'<|open|>message role="assistant"<|sep|><|open|>{channel}<|sep|>'
        return text

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        thinking=True,
        thinking_effort="max",
    ):
        rendered = self._render(messages, tools, thinking, thinking_effort, add_generation_prompt)
        return [ord(char) for char in rendered] if tokenize else rendered


def _decode(ids):
    return "".join(chr(token_id) for token_id in ids)


def _input_ids(value):
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def test_kimi_k3_inputs_match_tokenizer_and_only_assistant_is_supervised():
    tokenizer = FakeKimiK3Tokenizer()
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="kimi-k3").build(
        tokenizer,
        tokenizer_hash="fake-kimi-k3",
    )
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    messages = [
        {"role": "system", "content": "Follow instructions."},
        {"role": "user", "content": "Question with <|open|> as ordinary text."},
        {"role": "assistant", "reasoning_content": "Think.", "content": "Answer one."},
        {"role": "user", "content": "Question two."},
        {"role": "assistant", "content": "Do not train this.", "loss": False},
    ]

    tokenized = tokenize_fn({"messages": messages, "tools": tools})
    expected_ids = tokenizer.apply_chat_template(
        [{key: value for key, value in message.items() if key != "loss"} for message in messages],
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        thinking=True,
        thinking_effort="max",
    )

    assert tokenized["input_ids"] == expected_ids
    assert len(tokenized["input_ids"]) == len(tokenized["labels"])

    supervised = _decode([label for label in tokenized["labels"] if label != IGNORE_INDEX])
    assistant = tokenizer.apply_chat_template(
        [messages[2]],
        tokenize=False,
        add_generation_prompt=False,
        thinking=True,
        thinking_effort=None,
    )
    generation_prefix = tokenizer.apply_chat_template(
        [],
        tokenize=False,
        add_generation_prompt=True,
        thinking=True,
        thinking_effort=None,
    )
    assert assistant.startswith(generation_prefix)
    assert supervised == assistant[len(generation_prefix) :]
    assert supervised.startswith("Think.<|close|>think<|sep|>")
    assert "Follow instructions." not in supervised
    assert "Question with" not in supervised
    assert "Do not train this." not in supervised


def test_kimi_k3_generation_prompt_is_masked():
    tokenizer = FakeKimiK3Tokenizer()
    from xtuner.v1.data_proto.messages import KimiK3ChatMessages
    from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP

    tokenized = KimiK3ChatMessages(messages=[{"role": "user", "content": "Question."}]).tokenize(
        tokenizer,
        CHAT_TEMPLATE_MAP["kimi-k3"],
        add_generation_prompt=True,
    )

    assert _decode(tokenized["input_ids"]).endswith('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
    assert all(label == IGNORE_INDEX for label in tokenized["labels"])


@pytest.mark.skipif(not os.getenv("KIMI_K3_PATH"), reason="KIMI_K3_PATH is not set")
def test_kimi_k3_matches_official_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["KIMI_K3_PATH"], trust_remote_code=True)
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="kimi-k3").build(tokenizer)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Treat <|open|> as user text."},
        {"role": "assistant", "reasoning_content": "Reasoning.", "content": "Answer."},
    ]

    tokenized = tokenize_fn({"messages": messages})
    expected = _input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            thinking=True,
            thinking_effort="max",
        )
    )
    assistant = _input_ids(
        tokenizer.apply_chat_template(
            [messages[-1]],
            tokenize=True,
            add_generation_prompt=False,
            thinking=True,
            thinking_effort=None,
        )
    )
    generation_prefix = _input_ids(
        tokenizer.apply_chat_template(
            [],
            tokenize=True,
            add_generation_prompt=True,
            thinking=True,
            thinking_effort=None,
        )
    )

    assert tokenized["input_ids"] == expected
    assert assistant[: len(generation_prefix)] == generation_prefix
    assert [label for label in tokenized["labels"] if label != IGNORE_INDEX] == assistant[len(generation_prefix) :]
