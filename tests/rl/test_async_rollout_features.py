import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop
from xtuner.v1.rl.agent_loop.producer import AsyncProduceStrategy
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.utils import get_logger


logger = get_logger()


class DummyAgentLoop(AgentLoop):
    def __init__(self, max_tokens: int = 6, eos_tokens: list[int] | None = None):
        # test double: avoid heavy AgentLoop.__init__ dependencies
        self.max_tokens = max_tokens
        self.eos_tokens = eos_tokens or [2]
        self.logger = logger
        self.rollout_ctl = object()
        self.sample_params = SampleParams(max_tokens=max_tokens)
        self.infer_engine_calls = 0

    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        # emulate rollout-worker short-circuit behavior:
        # if existing response already has EOS or reaches max tokens, skip engine call
        current_ids = list(rollout_state.response_ids or [])
        hit_eos = any(token in self.eos_tokens for token in current_ids)
        hit_max = len(current_ids) >= self.max_tokens
        if hit_eos or hit_max:
            rollout_state.status = Status.COMPLETED
            return rollout_state

        # engine is called only when not short-circuited
        self.infer_engine_calls += 1
        rollout_state.response_ids = [9]
        rollout_state.response = "n"
        rollout_state.logprobs = [0.5]
        rollout_state.response_mask = [1]
        rollout_state.status = Status.COMPLETED
        return rollout_state


class InstrumentedSampler:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.next_uid = 0
        self.sampled_aborted_uids: list[int] = []

    async def sample(self, task_name: str, sample_from_expired_storage: bool = False):
        aborted = await self.replay_buffer.get(1, task_name=task_name, group_status=Status.ABORTED)
        if aborted:
            uid = aborted[0][0].uid
            self.sampled_aborted_uids.append(uid)
            return aborted[0]

        uid = self.next_uid
        self.next_uid += 1
        return [
            RolloutState(
                uid=uid,
                message=[{"role": "user", "content": f"q-{uid}"}],
                prompt_ids=[1],
                status=Status.INIT,
                response_ids=[],
            )
        ]


class MultiRoundAgentLoop:
    def __init__(self):
        self.rollout_ctl = object()
        self.round_id = 1
        self.generated_total = 0

    async def generate_group(self, rollout_state, rollout_step=0, enable_partial_rollout=False):
        self.generated_total += len(rollout_state)
        uid = rollout_state[0].uid
        if self.round_id == 1:
            if uid in {4, 5}:
                await asyncio.sleep(0.05)
                rollout_state[0].status = Status.ABORTED
            else:
                await asyncio.sleep(0.005)
                rollout_state[0].status = Status.COMPLETED
        else:
            await asyncio.sleep(0.005)
            rollout_state[0].status = Status.COMPLETED
        return rollout_state


class TestAsyncRolloutFeatures(unittest.IsolatedAsyncioTestCase):
    async def test_tail_batch_mark_expired_thresholds(self):
        def make_sample(staleness: int) -> RolloutState:
            return RolloutState(
                uid=100 + staleness,
                message=[{"role": "user", "content": "tail"}],
                prompt_ids=[1],
                response_ids=[1],
                status=Status.ABORTED,
                seq_staleness=staleness,
            )

        strategy_t1 = AsyncProduceStrategy(
            over_sample_threshold=0.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=0,
            tail_batch_stale_threshold=1,
            is_valid_sample_fn=lambda samples: True,
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )
        samples_t1 = [make_sample(1), make_sample(2)]
        strategy_t1.mark_expired_samples(samples_t1)
        self.assertEqual(samples_t1[0].status, Status.ABORTED)
        self.assertEqual(samples_t1[1].status, Status.EXPIRED)

        strategy_t4 = AsyncProduceStrategy(
            over_sample_threshold=0.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=0,
            tail_batch_stale_threshold=4,
            is_valid_sample_fn=lambda samples: True,
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )
        samples_t4 = [make_sample(4), make_sample(5)]
        strategy_t4.mark_expired_samples(samples_t4)
        self.assertEqual(samples_t4[0].status, Status.ABORTED)
        self.assertEqual(samples_t4[1].status, Status.EXPIRED)

    async def test_tail_batch_mode_switch_and_reset_staleness(self):
        class TailSampler:
            def __init__(self, replay_buffer):
                self.replay_buffer = replay_buffer
                self.flags: list[bool] = []
                self.uid = 0

            async def sample(self, task_name: str, sample_from_expired_storage: bool = False):
                self.flags.append(sample_from_expired_storage)
                if sample_from_expired_storage:
                    expired = await self.replay_buffer.get(1, task_name=task_name, group_status=Status.EXPIRED)
                    if expired:
                        for item in expired[0]:
                            item.seq_staleness = 0
                        return expired[0]
                item = RolloutState(
                    uid=1000 + self.uid,
                    message=[{"role": "user", "content": "fresh"}],
                    prompt_ids=[1],
                    response_ids=[],
                    status=Status.INIT,
                    seq_staleness=0,
                )
                self.uid += 1
                return [item]

        class TailAgentLoop:
            def __init__(self):
                self.rollout_ctl = object()

            async def generate_group(self, rollout_state, rollout_step=0, enable_partial_rollout=False):
                for item in rollout_state:
                    item.status = Status.COMPLETED
                return rollout_state

        replay_buffer = AsyncReplayBufferConfig(min_staleness=1, max_staleness=5).build()
        task_name = "tail_mode_task"
        # put two expired groups so expired_count > trigger_size(=1)
        for i in range(2):
            await replay_buffer.put(
                [
                    RolloutState(
                        uid=200 + i,
                        message=[{"role": "user", "content": "expired"}],
                        prompt_ids=[1],
                        response_ids=[9],
                        status=Status.EXPIRED,
                        seq_staleness=6,
                    )
                ],
                task_name,
            )

        sampler = TailSampler(replay_buffer)
        agent_loop = TailAgentLoop()
        strategy = AsyncProduceStrategy(
            over_sample_threshold=1.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=1,
            tail_batch_stale_threshold=1,
            is_valid_sample_fn=lambda samples: all(s.status == Status.COMPLETED for s in samples),
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )

        with (
            patch("xtuner.v1.rl.agent_loop.producer.continue_genertion", new=AsyncMock()),
            patch("xtuner.v1.rl.agent_loop.producer.pause_generation", new=AsyncMock()),
        ):
            await strategy.produce_batch(agent_loop, sampler, replay_buffer, batch_size=1, task_name=task_name, rollout_step=1)

        self.assertTrue(any(sampler.flags))
        completed_groups = await replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertGreater(len(completed_groups), 0)
        self.assertEqual(completed_groups[0][0].seq_staleness, 0)

    async def test_oversampling_round_consistency(self):
        replay_buffer = AsyncReplayBufferConfig(min_staleness=1, max_staleness=5).build()
        sampler = InstrumentedSampler(replay_buffer)
        agent_loop = MultiRoundAgentLoop()
        strategy = AsyncProduceStrategy(
            over_sample_threshold=0.5,
            enable_partial_rollout=False,
            tail_batch_trigger_size=0,
            tail_batch_stale_threshold=0,
            is_valid_sample_fn=lambda samples: all(s.status == Status.COMPLETED for s in samples),
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )

        task_name = "oversample_task"
        batch_size = 4
        initial_data_concurrency = int((1 + strategy.over_sample_threshold) * batch_size)

        with (
            patch("xtuner.v1.rl.agent_loop.producer.continue_genertion", new=AsyncMock()),
            patch("xtuner.v1.rl.agent_loop.producer.pause_generation", new=AsyncMock()),
        ):
            await strategy.produce_batch(agent_loop, sampler, replay_buffer, batch_size, task_name, rollout_step=0)

        remain_aborted = await replay_buffer.count(task_name=task_name, group_status=Status.ABORTED)
        remain_completed = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)

        # 1.1: round-1 remain(ABORTED + COMPLETED) equals initial data concurrency
        self.assertEqual(remain_aborted + remain_completed, initial_data_concurrency)

        # extra check from your question-1: total generated samples equals initial data concurrency
        self.assertEqual(agent_loop.generated_total, initial_data_concurrency)

        # 1.2: round-2 should sample aborted items exactly equal to previous remaining aborted
        last_round_remaining = remain_aborted
        agent_loop.round_id = 2
        with (
            patch("xtuner.v1.rl.agent_loop.producer.continue_genertion", new=AsyncMock()),
            patch("xtuner.v1.rl.agent_loop.producer.pause_generation", new=AsyncMock()),
        ):
            await strategy.produce_batch(agent_loop, sampler, replay_buffer, batch_size, task_name, rollout_step=1)

        self.assertEqual(len(sampler.sampled_aborted_uids), last_round_remaining)

    async def test_partial_rollout_behavior(self):
        loop = DummyAgentLoop(max_tokens=5, eos_tokens=[2])

        # 2.1 same sample concatenation across rollouts
        state = RolloutState(
            uid=1,
            message=[{"role": "user", "content": "hello"}],
            prompt_ids=[10, 11],
            response_ids=[3, 4],
            response="ab",
            logprobs=[0.1, 0.2],
            response_mask=[1, 1],
            status=Status.ABORTED,
            extra_fields={},
        )
        out = await loop._generate_pipeline(state, rollout_step=3, enable_partial_rollout=True)
        self.assertEqual(out.response_ids, [3, 4, 9])
        self.assertEqual(loop.infer_engine_calls, 1)

        # 2.2 EOS in previous output -> engine not called + response_ids unchanged
        eos_state = RolloutState(
            uid=2,
            message=[{"role": "user", "content": "hi"}],
            prompt_ids=[10],
            response_ids=[7, 2],
            response="done",
            status=Status.ABORTED,
            extra_fields={},
        )
        before_calls = loop.infer_engine_calls
        eos_out = await loop._generate_pipeline(eos_state, rollout_step=4, enable_partial_rollout=True)
        self.assertEqual(eos_out.response_ids, [7, 2])
        self.assertEqual(loop.infer_engine_calls, before_calls)

        # 2.3 previous output reaches/exceeds max tokens -> engine not called + response_ids unchanged
        long_state = RolloutState(
            uid=3,
            message=[{"role": "user", "content": "long"}],
            prompt_ids=[10],
            response_ids=[1, 3, 4, 5, 6],
            response="xxxxx",
            status=Status.ABORTED,
            extra_fields={},
        )
        before_calls = loop.infer_engine_calls
        long_out = await loop._generate_pipeline(long_state, rollout_step=5, enable_partial_rollout=True)
        self.assertEqual(long_out.response_ids, [1, 3, 4, 5, 6])
        self.assertEqual(loop.infer_engine_calls, before_calls)

        # 2.4 multi-round partial rollout keeps response_ids length <= total max tokens
        multi = RolloutState(
            uid=4,
            message=[{"role": "user", "content": "m"}],
            prompt_ids=[10],
            response_ids=[8, 8],
            response="aa",
            status=Status.ABORTED,
            extra_fields={},
        )
        for step in range(6, 12):
            multi.status = Status.ABORTED
            multi = await loop._generate_pipeline(multi, rollout_step=step, enable_partial_rollout=True)
        self.assertLessEqual(len(multi.response_ids), loop.max_tokens)


if __name__ == "__main__":
    unittest.main()
