import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig


class MockState:
    def __init__(self, id, staleness=0, input_ids=None, status=Status.COMPLETED):
        self.id = id
        self.seq_staleness = staleness
        self.status = status
        self.input_ids = input_ids if input_ids is not None else [id]


class TestReplayBuffer(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _get_sorted_input_ids(data_groups):
        return sorted(tuple(state.input_ids) for group in data_groups for state in group)

    async def test_fifo_backend(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        group_states1 = [MockState(i) for i in range(1, 4)]
        group_states2 = [MockState(i) for i in range(5, 7)]

        await buffer.put(group_states1, "task1")
        await buffer.put(group_states2, "task1")
        res = await buffer.get(2, "task1", Status.COMPLETED)

        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 2)
        self.assertEqual(res[0][0].id, 1)
        self.assertEqual(res[1][0].id, 5)

    async def test_staleness_priority(self):
        replay_buffer_cfg = AsyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()

        s1 = MockState(id="low", staleness=1)
        s5 = MockState(id="high", staleness=5)

        await buffer.put([s1], "task1")
        await buffer.put([s5], "task1")

        res = await buffer.get(2, "task1", Status.COMPLETED)
        self.assertEqual(res[0][0].id, "high")
        self.assertEqual(res[1][0].id, "low")

    async def test_multi_task(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        await buffer.put([MockState(100)], "task_a")
        await buffer.put([MockState(200)], "task_b")

        res_a = await buffer.get(10, "task_a", Status.COMPLETED)
        res_b = await buffer.get(10, "task_b", Status.COMPLETED)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(len(res_a[0]), 1)
        self.assertEqual(res_a[0][0].id, 100)
        self.assertEqual(len(res_b), 1)
        self.assertEqual(len(res_b[0]), 1)
        self.assertEqual(res_b[0][0].id, 200)

    async def test_save_resume_fifo_backend(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        await buffer.put([MockState(1, input_ids=[11, 12]), MockState(2, input_ids=[21, 22])], "task_a")
        await buffer.put([MockState(3, input_ids=[31, 32])], "task_b")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_fifo.pkl"
            await buffer.save(save_path)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)

            self.assertEqual(await resumed_buffer.count("task_a", Status.COMPLETED), 1)
            self.assertEqual(await resumed_buffer.count("task_b", Status.COMPLETED), 1)

            resumed_task_a = await resumed_buffer.get(1, "task_a", Status.COMPLETED)
            resumed_task_b = await resumed_buffer.get(1, "task_b", Status.COMPLETED)
            self.assertEqual([state.id for state in resumed_task_a[0]], [1, 2])
            self.assertEqual([state.id for state in resumed_task_b[0]], [3])

            await resumed_buffer.put([MockState(4)], "task_a")
            resumed_task_a_next = await resumed_buffer.get(1, "task_a", Status.COMPLETED)
            self.assertEqual([state.id for state in resumed_task_a_next[0]], [4])

    async def test_save_resume_sample_keeps_input_ids(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        original_buffer = replay_buffer_cfg.build()
        await original_buffer.put([MockState(1, input_ids=[101, 102]), MockState(2, input_ids=[201])], "task")
        await original_buffer.put([MockState(3, input_ids=[301, 302, 303])], "task")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_input_ids.pkl"
            await original_buffer.save(save_path)

            old_sampled = await original_buffer.get(2, "task", Status.COMPLETED)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)
            new_sampled = await resumed_buffer.get(2, "task", Status.COMPLETED)

            ids_old = self._get_sorted_input_ids(old_sampled)
            ids_new = self._get_sorted_input_ids(new_sampled)
            self.assertEqual(ids_old, ids_new)

    async def test_save_resume_staleness_backend(self):
        replay_buffer_cfg = AsyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        await buffer.put([MockState("low", staleness=1)], "task")
        await buffer.put([MockState("high", staleness=5)], "task")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_staleness.pkl"
            await buffer.save(save_path)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)

            resumed = await resumed_buffer.get(2, "task", Status.COMPLETED)
            self.assertEqual(resumed[0][0].id, "high")
            self.assertEqual(resumed[1][0].id, "low")

    async def test_save_resume_sample_keeps_input_ids_staleness_backend(self):
        replay_buffer_cfg = AsyncReplayBufferConfig()
        original_buffer = replay_buffer_cfg.build()
        await original_buffer.put([MockState("mid", staleness=3, input_ids=[301, 302])], "task")
        await original_buffer.put([MockState("high", staleness=5, input_ids=[501])], "task")
        await original_buffer.put([MockState("low", staleness=1, input_ids=[101, 102, 103])], "task")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_staleness_input_ids.pkl"
            await original_buffer.save(save_path)

            old_sampled = await original_buffer.get(3, "task", Status.COMPLETED)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)
            new_sampled = await resumed_buffer.get(3, "task", Status.COMPLETED)

            ids_old = self._get_sorted_input_ids(old_sampled)
            ids_new = self._get_sorted_input_ids(new_sampled)
            self.assertEqual(ids_old, ids_new)

    async def test_resume_keeps_fifo_query_filtering(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        await buffer.put([MockState("a1", status=Status.COMPLETED)], "task_a")
        await buffer.put([MockState("a2", status=Status.FAILED)], "task_a")
        await buffer.put([MockState("b1", status=Status.COMPLETED)], "task_b")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_fifo_query.pkl"
            await buffer.save(save_path)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)

            self.assertEqual(await resumed_buffer.count("task_a", Status.COMPLETED), 1)
            self.assertEqual(await resumed_buffer.count("task_a", Status.FAILED), 1)
            self.assertEqual(await resumed_buffer.count("task_b", Status.COMPLETED), 1)
            self.assertEqual(await resumed_buffer.count("task_b", Status.FAILED), 0)

            task_a_completed = await resumed_buffer.get(5, "task_a", Status.COMPLETED)
            self.assertEqual([s.id for s in task_a_completed[0]], ["a1"])

            task_a_failed = await resumed_buffer.get(5, "task_a", Status.FAILED)
            self.assertEqual([s.id for s in task_a_failed[0]], ["a2"])

    async def test_resume_keeps_staleness_query_filtering_and_order(self):
        replay_buffer_cfg = AsyncReplayBufferConfig()
        buffer = replay_buffer_cfg.build()
        await buffer.put([MockState("done_low", staleness=1, status=Status.COMPLETED)], "task")
        await buffer.put([MockState("failed_high", staleness=10, status=Status.FAILED)], "task")
        await buffer.put([MockState("done_mid", staleness=5, status=Status.COMPLETED)], "task")

        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "replay_buffer_staleness_query.pkl"
            await buffer.save(save_path)

            resumed_buffer = replay_buffer_cfg.build()
            await resumed_buffer.resume(save_path)

            self.assertEqual(await resumed_buffer.count("task", Status.COMPLETED), 2)
            self.assertEqual(await resumed_buffer.count("task", Status.FAILED), 1)

            completed = await resumed_buffer.get(2, "task", Status.COMPLETED)
            self.assertEqual(completed[0][0].id, "done_mid")
            self.assertEqual(completed[1][0].id, "done_low")

            failed = await resumed_buffer.get(1, "task", Status.FAILED)
            self.assertEqual(failed[0][0].id, "failed_high")
