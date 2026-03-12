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

    async def _preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        # for partial rollout
        if (
            not enable_partial_rollout
            or rollout_state.response_ids is None
            or len(rollout_state.response_ids) == 0
            or rollout_state.status == Status.COMPLETED
        ):
            return rollout_state

        response_ids = (
            rollout_state.response_ids.tolist()
            if hasattr(rollout_state.response_ids, "tolist")
            else list(rollout_state.response_ids)
        )
        rollout_state.response_ids = response_ids  # 之前n轮的response_ids
        response_len = len(rollout_state.response_ids)
        rollout_state.tokens = list(rollout_state.prompt_ids or []) + rollout_state.response_ids
        remaining_tokens = self.max_tokens - len(rollout_state.response_ids)
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})
        self.logger.info(
            f"Sample: {rollout_state.uid} continue rollout with {response_len} response tokens. Remaining tokens allowed: {remaining_tokens}. status: {rollout_state.status}, prompt_ids_len: {len(rollout_state.prompt_ids or [])}, response_ids_len: {len(rollout_state.response_ids)}, total_tokens_len: {len(rollout_state.tokens)}"
        )
        history_response_ids = (
            rollout_state.tokens[len(rollout_state.prompt_ids or []) :] if rollout_state.tokens else []
        )
        history_response = rollout_state.response if (enable_partial_rollout and rollout_state.response) else ""
        history_logprobs = rollout_state.logprobs if (enable_partial_rollout and rollout_state.logprobs) else []
        history_response_mask = (
            rollout_state.response_mask if (enable_partial_rollout and rollout_state.response_mask) else []
        )
        # TODO: 处理 routed_experts
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": history_response_ids,
            "response": history_response,
            "logprobs": history_logprobs,
            "response_mask": history_response_mask,
        }
        return rollout_state

    async def _postprocess(self, rollout_state: RolloutState, rollout_step: int) -> RolloutState:
        new_response_len = len(rollout_state.response_ids or [])
        history_response_dict = rollout_state.extra_fields.pop("history_response_dict", {})

        history_response_ids = history_response_dict.get("response_ids", [])
        history_logprobs = history_response_dict.get("logprobs", [])
        history_response_mask = history_response_dict.get("response_mask", [])
        history_response = history_response_dict.get("response", "")
        rollout_state.response_ids = history_response_ids + (rollout_state.response_ids or [])
        rollout_state.response = history_response + (rollout_state.response or "")
        rollout_state.logprobs = history_logprobs + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_response_mask + (rollout_state.response_mask or [])
        # TODO: 处理 routed_experts
        response_steps = [rollout_step] * new_response_len
        rollout_state.response_steps = (rollout_state.response_steps or []) + response_steps

        cur_rollout_steps = min(rollout_state.response_steps, default=rollout_step)
        rollout_state.seq_staleness = rollout_step - cur_rollout_steps

        return rollout_state

    async def _generate_pipeline(
        self, rollout_state: RolloutState, rollout_step: int = 0, enable_partial_rollout: bool = False
    ) -> RolloutState:
        rollout_state = await self._preprocess(rollout_state, enable_partial_rollout)  # preprocess for partial rollout
        rollout_state = await self.generate_sample(rollout_state)
        rollout_state = await self._postprocess(rollout_state, rollout_step)  # postprocess for partial rollout
        return rollout_state

    async def generate_group(
        self, rollout_state: list[RolloutState], rollout_step: int = 0, enable_partial_rollout: bool = False
    ) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self._generate_pipeline(state, rollout_step, enable_partial_rollout))
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
