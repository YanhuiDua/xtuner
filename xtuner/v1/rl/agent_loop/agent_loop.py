import asyncio
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import NativeJudger, RouterJudger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task, get_eos_token
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)  # TODO: extra="forbid"
    hf_checkpoint: str
    sample_params: SampleParams

    @abstractmethod
    def build(self, rollout_controller, judger=None, logger=None) -> "AgentLoop": ...


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    def build(self, rollout_controller, judger=None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            logger=logger,
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Callable | NativeJudger | RouterJudger | None = None,
        logger=None,
    ) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judger = judger
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
        self.max_tokens = self.sample_params.max_tokens
        eos_token = get_eos_token(self.hf_checkpoint)
        self.eos_tokens: list[int] = [eos_token] if isinstance(eos_token, int) else eos_token

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState: ...

    async def pause(self) -> None:
        """Pause the agent loop if supported by the implementation."""
        # Default implementation is a no-op to keep behavior unchanged.
        return None

    async def _preprocess(self, rollout_state: RolloutState) -> RolloutState:
        # for partial rollout
        if rollout_state.response_ids:
            is_completed = False
            response_len = len(rollout_state.response_ids)
            if response_len > self.max_tokens:
                self.logger.warning(
                    f"Response tokens exceed max_tokens limit: {response_len} > {self.max_tokens}. Truncating."
                )
                rollout_state.response_ids = rollout_state.response_ids[:self.max_tokens]
                if rollout_state.logprobs is not None:
                    rollout_state.logprobs = rollout_state.logprobs[:self.max_tokens]
                if rollout_state.response_mask is not None:
                    rollout_state.response_mask = rollout_state.response_mask[:self.max_tokens]
                if rollout_state.response_steps is not None:
                    rollout_state.response_steps = rollout_state.response_steps[:self.max_tokens]
                rollout_state.finish_reason = "length"
                is_completed = True
            elif response_len == self.max_tokens:
                rollout_state.finish_reason = "length"
                is_completed = True
            elif rollout_state.response_ids[-1] in self.eos_tokens:  # 修复属性名拼写匹配 __init__ 中的定义
                self.logger.warning("Response tokens end with EOS token. Marking rollout as completed.")
                rollout_state.finish_reason = "stop"
                is_completed = True

            # 命中截断或 EOS 提前结束
            if is_completed:
                rollout_state.status = Status.COMPLETED
                return await self.judge_sample(rollout_state)

            # 续写逻辑
            rollout_state.tokens = list(rollout_state.prompt_ids or []) + rollout_state.response_ids
            remaining_tokens = self.max_tokens - len(rollout_state.response_ids)
            rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        history_response_ids = (
            rollout_state.tokens[len(rollout_state.prompt_ids or []) :] if rollout_state.tokens else []
        )
        history_response = rollout_state.response or ""
        history_logprobs = rollout_state.logprobs or []
        history_response_mask = rollout_state.response_mask or []
        history_routed_experts = rollout_state.routed_experts or []
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": history_response_ids,
            "response": history_response,
            "logprobs": history_logprobs,
            "response_mask": history_response_mask,
            "routed_experts": history_routed_experts,
        }
        return rollout_state

    async def _postprocess(
        self, rollout_state: RolloutState, rollout_step: int, enable_partial_rollout: bool
    ) -> RolloutState:
        new_response_len = len(rollout_state.response_ids or [])
        history_response_dict = rollout_state.extra_fields.pop("history_response_dict", {})
        rollout_state.response_ids = history_response_dict.get("response_ids", []) + (rollout_state.response_ids or [])
        rollout_state.response = history_response_dict.get("response", "") + (rollout_state.response or "")
        rollout_state.logprobs = history_response_dict.get("logprobs", []) + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_response_dict.get("response_mask", []) + (rollout_state.response_mask or [])
        rollout_state.routed_experts = history_response_dict.get("routed_experts", []) + (
            rollout_state.routed_experts or []
        )

        if rollout_state.status == Status.ABORTED:
            response_steps = [rollout_step] * new_response_len
            if rollout_state.response_steps is None:
                rollout_state.response_steps = response_steps
            else:
                rollout_state.response_steps = rollout_state.response_steps + response_steps
            if not enable_partial_rollout:
                rollout_state.clear_response()

        if not rollout_state.response_steps:
            rollout_state.seq_staleness = 0
        else:
            valid_steps = rollout_state.response_steps
            if rollout_state.response_mask is not None and len(rollout_state.response_mask) == len(
                rollout_state.response_steps
            ):
                unmasked_steps = [
                    step for step, mask in zip(rollout_state.response_steps, rollout_state.response_mask) if mask == 1
                ]
                if unmasked_steps:
                    valid_steps = unmasked_steps
            rollout_state.seq_staleness = max(0, rollout_step - min(valid_steps))
        return rollout_state

    async def _run_generation_pipeline(
        self, rollout_state: RolloutState, rollout_step: int = 0, enable_partial_rollout: bool = False
    ) -> RolloutState:
        rollout_state = await self._preprocess(rollout_state)
        if rollout_state.status == Status.COMPLETED:
            rollout_state.extra_fields.pop("history_response_dict", None)
            if not rollout_state.response_steps:
                rollout_state.seq_staleness = 0
            else:
                valid_steps = rollout_state.response_steps
                if rollout_state.response_mask is not None and len(rollout_state.response_mask) == len(
                    rollout_state.response_steps
                ):
                    unmasked_steps = [
                        step for step, mask in zip(rollout_state.response_steps, rollout_state.response_mask) if mask == 1
                    ]
                    if unmasked_steps:
                        valid_steps = unmasked_steps
                rollout_state.seq_staleness = max(0, rollout_step - min(valid_steps))
            return rollout_state

        rollout_state = await self.generate_sample(rollout_state)
        rollout_state = await self._postprocess(rollout_state, rollout_step, enable_partial_rollout)
        return rollout_state

    async def generate_group(
        self, rollout_state: list[RolloutState], rollout_step: int = 0, enable_partial_rollout: bool = False
    ) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self._run_generation_pipeline(state, rollout_step, enable_partial_rollout))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def judge_sample(self, rollout_state: RolloutState) -> RolloutState:
        if self.judger is None:
            return rollout_state
        if callable(self.judger):
            rollout_state = await self.judger(rollout_state)
        elif isinstance(self.judger, RouterJudger) or isinstance(self.judger, NativeJudger):
            rollout_state = await self.judger.judge(rollout_state)  # type: ignore[operator]
        else:
            raise ValueError(f"Invalid judger type: {type(self.judger)}")
        return rollout_state


class SingleTurnAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        assert rollout_state.sample_params is not None, "sample_params must be set in rollout_state"
        rollout_state.tokens = rollout_state.prompt_ids
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        rollout_state = await self.judge_sample(rollout_state)
        return rollout_state
