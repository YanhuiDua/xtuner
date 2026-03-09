import asyncio
import atexit
import importlib
import signal
import socket
import subprocess
import typing
from abc import ABC
from asyncio import AbstractEventLoop, Task
from typing import TYPE_CHECKING, Any, Callable, Coroutine, List, Literal, Optional, Union, cast

import ray
import torch.nn.functional as F

from xtuner.v1.utils.logger import get_logger


ScalarOperator = Literal["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"]
SetOperator = Literal["$in", "$not_in"]
BetweenOperator = Literal["$between"]
Operators = Union[ScalarOperator, SetOperator, BetweenOperator]
LogicOperator = Literal["$and", "$or"]


class QueryNode(ABC):
    """查询语法树的基类，仅作数据结构标记."""

    pass


class ConditionNode(QueryNode):
    """代表一个具体的查询条件."""

    field: str


class ScalarNode(ConditionNode):
    def __init__(self, field: str, op: ScalarOperator, value: Any):
        self.field = field
        self.op = op
        self.value = value


class SetNode(ConditionNode):
    def __init__(self, field: str, op: SetOperator, value: list[Any] | tuple[Any]):
        self.field = field
        self.op = op
        self.value = value


class BetweenNode(ConditionNode):
    def __init__(self, field: str, lower: Any, upper: Any):
        if lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        self.field = field
        self.op = "$between"
        self.lower = lower
        self.upper = upper


class LogicNode(QueryNode):
    """复合逻辑组."""

    def __init__(self, relation: LogicOperator, conditions: List[QueryNode]):
        self.relation = relation
        self.conditions = conditions


def parse_query(expr: Union[dict, QueryNode]) -> QueryNode:
    """将基于字典的 DSL 解析为纯粹的 AST 节点树 (ConditionNode, LogicNode)"""
    if isinstance(expr, QueryNode):
        return expr

    if isinstance(expr, dict):
        conditions: list[QueryNode] = []
        for key, value in expr.items():
            if key in ("$and", "$or"):
                if isinstance(value, list):
                    sub_asts = [parse_query(sub_expr) for sub_expr in value]
                    conditions.append(LogicNode(key, sub_asts))  # type: ignore
                else:
                    raise ValueError(f"逻辑操作符 {key} 的值必须是一个列表")
            else:
                if isinstance(value, dict):
                    # 例如: {"staleness": {"$lt": 5, "$gt": 0}}
                    for op, op_val in value.items():
                        if op in typing.get_args(ScalarOperator):
                            conditions.append(ScalarNode(field=key, op=op, value=op_val))
                        elif op in typing.get_args(SetOperator):
                            if not isinstance(op_val, (list, tuple)):
                                raise ValueError(f"操作符 '{op}' 需要传入一个列表或元组")
                            conditions.append(SetNode(field=key, op=op, value=op_val))
                        elif op == "$between":
                            if not isinstance(op_val, (list, tuple)) or len(op_val) != 2:
                                raise ValueError("操作符 '$between' 需要传入包含2个元素的列表或元组")
                            conditions.append(BetweenNode(field=key, lower=op_val[0], upper=op_val[1]))
                        else:
                            raise ValueError(f"不支持的操作符: {op}")
                else:
                    # 隐式等值，例如: {"task_name": "math"} -> "$eq"
                    conditions.append(ScalarNode(field=key, op="$eq", value=value))

        if len(conditions) > 1:
            # 默认多个条件之间是 AND 关系，例如: {"uid": "123", "status": {"$in": ["pending", "running]}}}
            return LogicNode("$and", conditions)  # type: ignore
        return conditions[0] if conditions else LogicNode("$and", [])

    raise ValueError(f"不支持的查询表达式格式: {expr}")


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs


logger = get_logger()


def close_ray():
    """Clean up the ray resource."""
    import ray

    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup():
    """Register cleanup handlers for Ray on exit and signals."""
    _cleaned = False

    def cleanup_once():
        nonlocal _cleaned
        if not _cleaned:
            _cleaned = True
            close_ray()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once()
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup_once)


if TYPE_CHECKING:
    import ray.actor

    from xtuner.v1.rl.utils.accelerator import AcceleratorType


def get_ray_accelerator() -> "AcceleratorType":
    from xtuner.v1.utils.device import get_device

    """Get the type of accelerator available in the Ray environment.

    This function checks for the availability of CUDA and NPU devices and
    returns the corresponding accelerator type.

    Returns:
        AcceleratorType: The type of accelerator ("GPU" or "NPU").

    Raises:
        NotImplementedError: If neither CUDA nor NPU is available.
    """
    accelerator = None
    if get_device() == "cuda":
        accelerator = "GPU"
        return "GPU"
    else:
        try:
            import torch_npu  # noqa: F401

            accelerator = "NPU"
        except ImportError:
            pass

    if accelerator is None:
        raise NotImplementedError(
            "Supports only CUDA or NPU. If your device is CUDA or NPU, "
            "please make sure that your environmental settings are "
            "configured correctly."
        )

    return cast("AcceleratorType", accelerator)


def load_function(path):
    """Load a function from a module.

    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _is_port_available(check_socket: socket.socket, port: int) -> bool:
    try:
        check_socket.bind(("", port))
        check_socket.listen(1)
        return True
    except OSError:
        return False


@ray.remote
def find_master_addr_and_port(nums=1, start_port=None, end_port=None):
    """Finds an available master address and a specified number of ports.

    This remote function gets the node's IP address and binds to one or more
    available ports, which can be used for distributed communication.

    Args:
        nums (int): The number of ports to find. Defaults to 1.
        start_port (Optional[int]): The starting port to search from.
            If None, random available ports will be used. Defaults to None.
        end_port (Optional[int]): The ending port to search to (exclusive).
            If start_port is None, this parameter is ignored. Defaults to None.

    Returns:
        A tuple containing the address and a single port if `nums` is 1,
        or a list of ports if `nums` is greater than 1.
    """
    addr = ray.util.get_node_ip_address()
    ports: list[int] = []
    sockets: list[socket.socket] = []

    if start_port is None:
        for _ in range(nums):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # if the port is binded and listened by this socket and then we close it,
            # socket.SO_REUSEADDR would make the port be reusable even it's in TIME_WAIT state.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sockets.append(s)
            if _is_port_available(check_socket=s, port=0):
                ports.append(s.getsockname()[1])
    else:
        assert isinstance(start_port, int), "If start_port isn't None, it must be an integer."
        assert isinstance(end_port, int), "If start_port isn't None, end_port must be an integer."
        assert end_port - start_port >= nums, (
            "If start_port isn't None, the range between start_port and end_port must be at least nums."
        )

        for candidate_port in range(start_port, end_port):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # if the port is binded and listened by this socket and then we close it,
            # socket.SO_REUSEADDR would make the port be reusable even it's in TIME_WAIT state.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sockets.append(s)
            if _is_port_available(check_socket=s, port=candidate_port):
                ports.append(candidate_port)
            # enough ports found
            if len(ports) >= nums:
                break

        if len(ports) < nums:
            raise RuntimeError(f"Could not find {nums} available ports starting from port {start_port} to {end_port}.")

    # close all sockets, no matter available or not
    for s in sockets:
        s.close()

    if len(ports) == 1:
        return addr, ports[0]
    else:
        return addr, ports


@ray.remote
def get_accelerator_ids(accelerator: str) -> list:
    """Get the IDs of the available accelerators (GPUs, NPUs, etc.) in the Ray
    cluster."""
    return ray.get_runtime_context().get_accelerator_ids()[accelerator]


def bind_train_rollout(
    train_workers,
    rollout_controller,
) -> None:
    """Bind the training and rollout workers for updating weights.

    This function retrieves rollout information from the rollout controller
    and distributes it to the training workers, enabling them to update the
    rollout models' weights.

    Args:
        train_workers: A list of training worker actors.
        rollout_controller: The rollout controller actor.
    """
    info_dict = ray.get(rollout_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get([worker.update_rollout_info.remote(**info_dict) for worker in train_workers])  # type: ignore[attr-defined]
    return


def handle_task_exception(task: Task):
    """Handles exceptions from an asyncio Task.

    This function checks if a task has raised an exception and, if so,
    re-raises it. It ignores `asyncio.CancelledError`.

    Args:
        task (Task): The asyncio task to check for exceptions.

    Raises:
        Exception: The exception raised by the task.
    """
    try:
        exc = task.exception()
        if exc is not None:
            raise exc
    except asyncio.CancelledError:
        pass  # Task was cancelled, ignore


def create_task(
    coro: Coroutine,
    loop: Optional[AbstractEventLoop] = None,
    done_callbacks: Optional[List[Callable[[Task], object]]] = None,
) -> asyncio.tasks.Task:
    """Creates and configures an asyncio Task.

    This function creates a task from a coroutine and attaches specified
    done callbacks. By default, it includes a callback to handle exceptions.

    Args:
        coro (Coroutine): The coroutine to wrap in a task.
        loop (Optional[AbstractEventLoop], optional): The event loop to run
            the task in. If None, the current event loop is used.
            Defaults to None.
        done_callbacks (Optional[List[Callable[[Task], object]]], optional):
            A list of callbacks to add to the task. If None, a default
            exception handler is used. Defaults to None.

    Returns:
        asyncio.tasks.Task: The created asyncio task.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    if done_callbacks is None:
        done_callbacks = [handle_task_exception]
    task = loop.create_task(coro)
    for callback in done_callbacks:
        task.add_done_callback(callback)
    return task
