import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoop
from .sampler import Sampler


logger = get_logger()


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


class ProduceStrategyConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(self) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    def build(self) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    tail_batch_stale_threshold: int = 0
    tail_batch_trigger_size: int = 0

    def build(self) -> "AsyncProduceStrategy":
        return AsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            tail_batch_stale_threshold=self.tail_batch_stale_threshold,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
    ): ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int,
    ):
        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(batch_size):
            rollout_state = await sampler.sample(task_name=task_name)
            task = create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        logger.info(f"Started {len(pending_tasks)} initial tasks for SyncProduceStrategy.")

        while self.should_continue_fn(completed_sample_count, batch_size):
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                items = task.result()
                if self.is_valid_sample_fn(items):
                    completed_sample_count += 1
                    logger.info(f"Collected {completed_sample_count}/{batch_size} valid samples for task {task_name}.")
                await replay_buffer.put(items, task_name)

            while len(pending_tasks) + completed_sample_count < batch_size and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        tail_batch_stale_threshold: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.tail_batch_stale_threshold = tail_batch_stale_threshold
        self.tail_batch_trigger_size = tail_batch_trigger_size

    def mark_expired_samples(self, samples: list[RolloutState], rollout_step: int) -> None:
        for sample in samples:
            if sample.status != Status.ABORTED or sample.response_steps is None or len(sample.response_steps) == 0:
                continue
            # 如果一个样本在 rollout_step 之前就被 aborted 了，并且距离现在已经超过 stale threshold 了，就认为它是过期的
            is_expired = rollout_step - min(sample.response_steps) > self.tail_batch_stale_threshold
            sample.status = Status.EXPIRED if is_expired else sample.status

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int,
    ):
        pending_tasks = set()
        init_completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        expired_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        sample_from_expired_storage = False
        data_concurrency = int((1 + self.over_sample_threshold) * batch_size)

        # 是否触发tail_batch
        if self.tail_batch_trigger_size > 0 and expired_sample_count >= self.tail_batch_trigger_size:
            logger.info(
                f"Tail batch trigger condition met: {expired_sample_count} expired samples (threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )
            sample_from_expired_storage = True
            data_concurrency = batch_size - init_completed_sample_count

        logger.info(
            f"Starting AsyncProduceStrategy with data concurrency: {data_concurrency}, initial completed samples: {init_completed_sample_count}, expired samples: {expired_sample_count}."
        )

        for _ in range(data_concurrency):
            rollout_state = await sampler.sample(
                task_name=task_name, sample_from_expired_storage=sample_from_expired_storage
            )
            task = create_task(
                agent_loop.generate_group(
                    rollout_state, rollout_step=rollout_step, enable_partial_rollout=self.enable_partial_rollout
                )
            )
            pending_tasks.add(task)

        completed_sample_count = init_completed_sample_count
        while self.should_continue_fn(completed_sample_count, batch_size):
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )

            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                items: list[RolloutState] = task.result()
                if self.is_valid_sample_fn(items):
                    completed_sample_count += 1
                self.mark_expired_samples(items, rollout_step)
                await replay_buffer.put(items, task_name)

            while len(
                pending_tasks
            ) + completed_sample_count < data_concurrency + init_completed_sample_count and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(
                    task_name=task_name, sample_from_expired_storage=sample_from_expired_storage
                )
                task = create_task(
                    agent_loop.generate_group(
                        rollout_state, rollout_step=rollout_step, enable_partial_rollout=self.enable_partial_rollout
                    )
                )
                pending_tasks.add(task)

        if len(pending_tasks) > 0:
            await agent_loop.pause()
            while len(pending_tasks) > 0:
                _, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                if len(pending_tasks) > 0:
                    await agent_loop.pause()
                    await asyncio.sleep(1)
        print("All worker tasks have completed after pausing env controller.")
