import re
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .native import NativeJudger


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> Tuple[bool, Optional[str]]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """

    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError:
        print("To use Math-Verify, please install it first by running `pip install math-verify`.")

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + solution_str + "}"

    try:
        answer = answer.split("</think>")[1].strip()
        ret_score, _ = verify_func([ground_truth_boxed], [answer])
    except Exception:
        pass
    except TimeoutException:
        ret_score = 0

    return ret_score==1, None # type: ignore[arg-type]


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {"score": reward, "acc": acc}


def compute_reward(response, label, extra_info):
    predict_str = response

    eos_token = extra_info["eos_token"]
    if isinstance(eos_token, list):
        for eos in eos_token:
            if response.endswith(eos):
                response = response[: -len(eos)]
                break
    else:
        if response.endswith(eos_token):
            response = response[: -len(eos_token)]

    out = compute_score(response, label)
    reward = out["score"]

    overlong_reward = 0
    if extra_info.get("enable_overlong_buffer", None):
        overlong_buffer_len = extra_info["overlong_buffer_len"]
        expected_len = extra_info["max_response_len"] - overlong_buffer_len
        valid_response_length = len(
            extra_info["tokenizer"](predict_str, return_tensors="pt")["input_ids"].flatten().tolist()
        )
        exceed_len = valid_response_length - expected_len
        overlong_penalty_factor = extra_info["overlong_penalty_factor"]
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
    return {"score": reward, "acc": out["acc"]}


class DapoMathJudgerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judger_name: str = "dapo_math"
    eos_token: List[str] | str
    enable_overlong_buffer: bool
    score: int = 1
    format_score: int = 0
    max_response_len: Optional[int] = None
    overlong_buffer_len: Optional[int] = None
    overlong_penalty_factor: Optional[float] = None
    tokenizer: Any = None
    extra_info: dict = Field(default_factory=dict)

    def __init__(
        self,
        judger_name: str,
        eos_token: List[str] | str,
        enable_overlong_buffer: bool,
        max_response_len: Optional[int],
        overlong_buffer_len: Optional[int],
        overlong_penalty_factor: Optional[float],
        tokenizer: Any,
        score: int = 1,
        format_score: int = 0,
        extra_info: dict = {},
    ):
        if isinstance(eos_token, str):
            assert eos_token.strip() != "", "eos_token string must not be empty"
        elif isinstance(eos_token, list):
            assert all(isinstance(e, str) and e.strip() != "" for e in eos_token), (
                "All eos_token list elements must be non-empty strings"
            )
            assert len(eos_token) > 0, "eos_token list must not be empty"
        else:
            raise TypeError("eos_token must be a non-empty string or a non-empty list of strings")

        # 初始化基类
        super().__init__(
            judger_name=judger_name,
            eos_token=eos_token,
            enable_overlong_buffer=enable_overlong_buffer,
            score=score,
            format_score=format_score,
            max_response_len=max_response_len,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
            tokenizer=tokenizer,
            extra_info=extra_info,
        )

        self.extra_info.update(
            {
                "eos_token": eos_token,
                "score": score,
                "format_score": format_score,
            }
        )

        if enable_overlong_buffer:
            assert max_response_len is not None
            assert overlong_buffer_len is not None
            assert overlong_penalty_factor is not None
            assert tokenizer is not None
            self.extra_info.update(
                {
                    "enable_overlong_buffer": enable_overlong_buffer,
                    "max_response_len": max_response_len,
                    "overlong_buffer_len": overlong_buffer_len,
                    "overlong_penalty_factor": overlong_penalty_factor,
                    "tokenizer": tokenizer,
                }
            )

    def build(self):
        return NativeJudger(judger_name=self.judger_name, reward_func=compute_reward, extra_info=self.extra_info)
