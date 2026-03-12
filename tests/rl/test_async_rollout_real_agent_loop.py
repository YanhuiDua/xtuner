from __future__ import annotations
import os
import unittest

import asyncio

import ray
import torch
from transformers import AutoTokenizer

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.agent_loop.producer import AsyncProduceStrategy
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig, get_eos_token
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers


class TestAsyncRolloutRealAgentLoop(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("XTUNER_USE_FA3", "1")
        os.environ.setdefault("LMD_SKIP_WARMUP", "1")

    def setUp(self):
        import torch

        ray.init(num_cpus=32, ignore_reinit_error=True)
        model_path = os.environ["ROLLOUT_MODEL_PATH"]
        accelerator_type = torch.accelerator.current_accelerator().type
        resource_map = {"npu": "NPU", "cuda": "GPU"}
        resources_cfg = AcceleratorResourcesConfig(
            accelerator=resource_map[accelerator_type],
            num_workers=1,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        rollout_config = RolloutConfig(
            env="test_async_rollout_real_agent_loop",
            model_path=model_path,
            model_name=os.path.basename(model_path).lower(),
            tokenizer_path=model_path,
            context_length=1536,
            rollout_max_batch_size_per_instance=16,
            max_retry_per_sample=0,
        )
        pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)
        self.rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.sample_params = SampleParams(max_tokens=64, temperature=0.0, return_token_ids=True)

    def tearDown(self):
        ray.shutdown()

    def _make_state(self, uid: int, prompt: str, max_tokens: int) -> RolloutState:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten().tolist()
        return RolloutState(
            uid=uid,
            message=[{"role": "user", "content": prompt}],
            prompt_ids=prompt_ids,
            sample_params=SampleParams(max_tokens=max_tokens, temperature=0.0, return_token_ids=True),
            status=Status.INIT,
            extra_fields={},
        )

    async def test_real_tail_batch_mark_expired_thresholds(self):
        def make_sample(staleness: int) -> RolloutState:
            return RolloutState(
                uid=500 + staleness,
                message=[{"role": "user", "content": "tail-threshold"}],
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
        sample_t1 = [make_sample(2)]
        strategy_t1.mark_expired_samples(sample_t1)
        self.assertEqual(sample_t1[0].status, Status.EXPIRED)

        strategy_t4 = AsyncProduceStrategy(
            over_sample_threshold=0.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=0,
            tail_batch_stale_threshold=4,
            is_valid_sample_fn=lambda samples: True,
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )
        sample_t4 = [make_sample(4)]
        strategy_t4.mark_expired_samples(sample_t4)
        self.assertEqual(sample_t4[0].status, Status.ABORTED)

    async def test_real_partial_rollout_short_circuit_keeps_response_ids(self):
        model_path = os.environ["ROLLOUT_MODEL_PATH"]
        agent_loop = SingleTurnAgentLoop(
            rollout_ctl=self.rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=model_path,
            judger=None,
        )

        # case 1: max_tokens exhausted in partial rollout -> response_ids should remain unchanged
        state_len = self._make_state(1, "Say hi", max_tokens=8)
        state_len.status = Status.ABORTED
        state_len.response_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        state_len.response = "abcdefgh"

        out_len = await agent_loop._generate_pipeline(state_len, rollout_step=1, enable_partial_rollout=True)
        self.assertEqual(out_len.response_ids, [11, 12, 13, 14, 15, 16, 17, 18])

        # case 2: EOS already reached in partial rollout -> response_ids should remain unchanged
        # EOS id comes from rollout-worker side source to ensure worker can perceive it.
        worker_eos = get_eos_token(model_path)
        eos_id = worker_eos[0] if isinstance(worker_eos, list) else worker_eos
        state_eos = self._make_state(2, "Say hi again", max_tokens=16)
        state_eos.status = Status.ABORTED
        state_eos.response_ids = [31, eos_id]
        state_eos.response = "done"

        out_eos = await agent_loop._generate_pipeline(state_eos, rollout_step=2, enable_partial_rollout=True)
        self.assertEqual(out_eos.response_ids, [31, eos_id])

        # case 3: after multiple partial rollouts, response_ids length should never exceed initial max_tokens
        total_max_tokens = 16
        multi = self._make_state(3, "Count from one.", max_tokens=total_max_tokens)
        multi.status = Status.ABORTED
        multi.response_ids = [41, 42]
        multi.response = "start"

        for step in range(3, 9):
            multi.status = Status.ABORTED
            multi = await agent_loop._generate_pipeline(multi, rollout_step=step, enable_partial_rollout=True)

        self.assertLessEqual(len(multi.response_ids), total_max_tokens)

    async def test_real_oversampling_round_consistency(self):
        class RealSampler:
            def __init__(self, replay_buffer, build_state_fn):
                self.replay_buffer = replay_buffer
                self.build_state_fn = build_state_fn
                self.uid = 0
                self.sampled_from_aborted = 0

            async def sample(self, task_name: str, sample_from_expired_storage: bool = False):
                aborted = await self.replay_buffer.get(1, task_name, Status.ABORTED)
                if aborted:
                    self.sampled_from_aborted += 1
                    return aborted[0]
                item = self.build_state_fn(self.uid)
                self.uid += 1
                return [item]

        class RealAgentLoopWrapper:
            def __init__(self, real_agent_loop):
                self._loop = real_agent_loop
                self.rollout_ctl = real_agent_loop.rollout_ctl
                self.generated_total = 0

            async def generate_group(self, rollout_state, rollout_step=0, enable_partial_rollout=False):
                self.generated_total += len(rollout_state)
                return await self._loop.generate_group(
                    rollout_state, rollout_step=rollout_step, enable_partial_rollout=enable_partial_rollout
                )

        model_path = os.environ["ROLLOUT_MODEL_PATH"]
        real_loop = SingleTurnAgentLoop(
            rollout_ctl=self.rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=model_path,
            judger=None,
        )
        loop = RealAgentLoopWrapper(real_loop)

        replay_buffer = AsyncReplayBufferConfig(min_staleness=1, max_staleness=5).build()
        sampler = RealSampler(
            replay_buffer,
            lambda uid: self._make_state(
                uid,
                "Write one short sentence." if uid % 2 == 0 else "Explain calculus in detail with many points.",
                max_tokens=32 if uid % 2 == 0 else 256,
            ),
        )

        strategy = AsyncProduceStrategy(
            over_sample_threshold=1.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=0,
            tail_batch_stale_threshold=0,
            is_valid_sample_fn=lambda samples: all(s.status == Status.COMPLETED for s in samples),
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )

        task_name = "real_oversampling_task"
        batch_size = 1
        initial_data_concurrency = int((1 + strategy.over_sample_threshold) * batch_size)

        await strategy.produce_batch(loop, sampler, replay_buffer, batch_size, task_name, rollout_step=0)

        remain_aborted = await replay_buffer.count(task_name=task_name, group_status=Status.ABORTED)
        remain_completed = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)

        self.assertEqual(remain_aborted + remain_completed, initial_data_concurrency)
        self.assertEqual(loop.generated_total, initial_data_concurrency)

        last_round_remaining = remain_aborted
        await strategy.produce_batch(loop, sampler, replay_buffer, batch_size, task_name, rollout_step=1)
        self.assertEqual(sampler.sampled_from_aborted, last_round_remaining)

    async def test_real_tail_batch_mode_switch_and_reset_staleness(self):
        class RealTailSampler:
            def __init__(self, replay_buffer):
                self.replay_buffer = replay_buffer
                self.sample_from_expired_flags: list[bool] = []
                self.uid = 0

            async def sample(self, task_name: str, sample_from_expired_storage: bool = False):
                self.sample_from_expired_flags.append(sample_from_expired_storage)
                if sample_from_expired_storage:
                    expired = await self.replay_buffer.get(1, task_name, Status.EXPIRED)
                    if expired:
                        for item in expired[0]:
                            item.seq_staleness = 0
                        return expired[0]
                state = RolloutState(
                    uid=800 + self.uid,
                    message=[{"role": "user", "content": "fallback"}],
                    prompt_ids=self.tokenizer("fallback", return_tensors="pt")["input_ids"].flatten().tolist(),
                    sample_params=SampleParams(max_tokens=8, temperature=0.0, return_token_ids=True),
                    status=Status.INIT,
                    extra_fields={},
                )
                self.uid += 1
                return [state]

        class RealLoopWrapper:
            def __init__(self, real_loop):
                self._real_loop = real_loop
                self.rollout_ctl = real_loop.rollout_ctl

            async def generate_group(self, rollout_state, rollout_step=0, enable_partial_rollout=False):
                for item in rollout_state:
                    item.status = Status.COMPLETED
                return rollout_state

        task_name = "real_tail_mode_task"
        replay_buffer = AsyncReplayBufferConfig(min_staleness=1, max_staleness=5).build()
        for i in range(2):
            await replay_buffer.put(
                [
                    RolloutState(
                        uid=900 + i,
                        message=[{"role": "user", "content": "expired"}],
                        prompt_ids=[1],
                        response_ids=[9],
                        status=Status.EXPIRED,
                        seq_staleness=7,
                    )
                ],
                task_name,
            )

        model_path = os.environ["ROLLOUT_MODEL_PATH"]
        real_loop = SingleTurnAgentLoop(
            rollout_ctl=self.rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=model_path,
            judger=None,
        )
        loop = RealLoopWrapper(real_loop)
        sampler = RealTailSampler(replay_buffer)

        strategy = AsyncProduceStrategy(
            over_sample_threshold=1.0,
            enable_partial_rollout=False,
            tail_batch_trigger_size=1,
            tail_batch_stale_threshold=1,
            is_valid_sample_fn=lambda samples: all(s.status == Status.COMPLETED for s in samples),
            should_continue_fn=lambda completed_count, batch_size, **kwargs: completed_count < batch_size,
        )

        await strategy.produce_batch(loop, sampler, replay_buffer, batch_size=1, task_name=task_name, rollout_step=3)

        self.assertTrue(any(sampler.sample_from_expired_flags))
        completed_groups = await replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertGreater(len(completed_groups), 0)
        self.assertEqual(completed_groups[0][0].seq_staleness, 0)


if __name__ == "__main__":
    unittest.main()
