from argparse import Namespace
from typing import Any, Dict, List, Union

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

from xtuner.v1.data_proto.rl_data import RolloutState, Status, update_status_from_finish_reason
from xtuner.v1.utils.device import get_device, get_torch_device_module

from .worker import RolloutConfig, RolloutWorker


def run_vllm_server_wrapper(server_args):
    uvloop.run(run_server(server_args))


class vLLMWorker(RolloutWorker):
    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        self.server_func = run_vllm_server_wrapper
        self.router_func = ""
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["output_ids"] = "output_ids"
        self.endpoints["response"] = "text"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wakeup"
        self.api_keys = self.config.api_key

    async def _create_request(
        self,
        url: str,
        prompt: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,  # reserved for agent tool use
        tool_choice: str,  # reserved for agent tool use
        sample_params: dict,
        extra_params: dict,
        extra_info: dict,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        payload = {
            "model": self.config.model_path,
            "messages": prompt,
            "stream": True,
        }
        payload.update(sample_params)
        payload.update(extra_params)

        return await self._safe_post_request(url, headers, payload)

    def get_logprobs(self, input_ids, sampling_params):
        pass

    def generate(self, input_ids, sampling_params):
        pass

    def sleep(self, level=1, tags: List[str] | None = None):
        import requests

        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, response.status_code
        return response.json()

    def wake_up(self, tags: List[str] | None = None):
        import requests

        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, response.status_code
        return response.json()

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def update_weights(self, ipc_handles):
        # todo
        pass

    def reset_prefix_cache(self):
        # todo
        pass

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        # use vllm FlexibleArgumentParser to parse the config
        # and return the args as the default server config
        # vllm server_args: vllm/vllm/engine/arg_utils.py
        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        args = parser.parse_args([])
        args.__dict__.update(vars(self.config))

        args = {}
        args["host"] = self.host
        args["port"] = self.server_port
        args["api_key"] = self.api_keys
        args["api_keys"] = self.api_keys
        args["model"] = self.config.model_path
        args["log_level"] = "info"
        args["data_parallel_size"] = self.dp_size
        args["tensor_parallel_size"] = self.tp_size
        args["enable_expert_parallel"] = False

        args["distributed_executor_backend"] = "ray"
        args["max_model_len"] = self.config.context_length
        args["enforce_eager"] = False
        args["enable_sleep_mode"] = True
        args["worker_extension_cls"] = "xtuner.v1.rl.rollout.vllm.WorkerWrap"
        args["trust_remote_code"] = True
        args["enable_prefix_caching"] = False
        args["allowed_local_media_path"] = "/"
        args["mm_processor_cache_gb"] = 0
        args["max_num_batched_tokens"] = 4096
        args["max_num_seqs"] = self.config.rollout_max_batch_size_per_instance // self.dp_size
        args["block_size"] = 128
        args["gpu_memory_utilization"] = self.config.gpu_memory_utilization
        args["compilation_config"] = {
            "cudagraph_capture_sizes": [16, 12, 8, 4, 2, 1],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
        args["additional_config"] = {"enable_cpu_binding": True}
        args["limit_mm_per_prompt"] = {"image": 10, "video": 0}
        args["enable_log_requests"] = False
        args["uvicorn_log_level"] = "error"
        env = {
            "VLLM_VERSION": "0.11.0",
            "TASK_QUEUE_ENABLE": "0",
            "CPU_AFFINITY_CONF": "2",
            "VLLM_USE_V1": "1",
            "VLLM_RAY_PER_WORKER_GPUS": "0.1",
            "VLLM_RAY_BUNDLE_INDICES": ",".join(map(str, self.engine_bundle_idxs)),
            "VLLM_MONITOR": "1",
            "VLLM_ACCU_MONITOR": "0",
            "CUSTOM_SCHEDULE_KV_LIMIT": "0.9",
            "HCCL_BUFFSIZE": "512",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "SHM_BARRIER": "true",
            "USE_TOKEN_IN": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "INTERNS1_VIT_USE_TP": "1",
            "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        }

        # Apply extra_rollout_config overrides for vLLM parameters (prefix: "vllm_")
        extra_cfg = getattr(self.config, "extra_rollout_config", None) or {}
        for key, value in extra_cfg.items():
            if key.startswith("vllm_"):
                real_key = key[5:]
                args[real_key] = value

        args_.__dict__.update(args)
        validate_parsed_serve_args(args_)

        return Namespace(
            args=args_,
            env=env,
            api_key=self.api_keys,
            api_keys=self.api_keys,
            ray_runtime_env={"env_vars": env},
        )

    async def _safe_handle_response(self, rollout_state: RolloutState, http_response) -> RolloutState:
        if rollout_state.sample_params.stream:
            return await self._handle_stream_response(rollout_state, http_response)
        return await self._handle_non_stream_response(rollout_state, http_response)

    async def _handle_stream_response(self, rollout_state: RolloutState, response) -> RolloutState:
        raise NotImplementedError

    async def _handle_non_stream_response(self, rollout_state: RolloutState, response) -> RolloutState:
        uid = rollout_state.uid or rollout_state.message_uid
        sample_params = rollout_state.sample_params
        last_token_ids: list[int] = []
        last_logprobs: list[float] = []
        routed_experts = None

        response_json = response.json()
        response_choice = response_json["choices"][0]
        if response_choice.get("logprobs") is not None:
            last_token_ids = response_choice.get("token_ids", response_json.get("token_ids", []))
            last_logprobs = [
                item["logprob"] for item in response_choice["logprobs"].get("content", []) if "logprob" in item
            ]
            assert len(last_token_ids) == len(last_logprobs)
            assert len(last_token_ids) <= sample_params.max_tokens, (
                f"Generation length exceeds limit: generated {len(last_token_ids)}, limit {sample_params.max_tokens}"
            )

        last_trajectory = response_choice["message"].get("content") or ""
        finish_reason = response_choice.get("finish_reason")
        if finish_reason == "abort" and self.receive_abort_request.is_set() is False:
            self.receive_abort_request.set()
            self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")

        if self.enable_return_routed_experts:
            routed_experts = response_choice.get("routed_experts", response_json.get("routed_experts"))
            if routed_experts is not None:
                if isinstance(routed_experts, str):
                    import base64

                    data = base64.b64decode(routed_experts)
                    routed_experts = ray.cloudpickle.loads(data)
                else:
                    routed_experts = torch.tensor(routed_experts)
                    routed_experts = ray.put(routed_experts)

        rollout_status = update_status_from_finish_reason(finish_reason)
        if rollout_status == Status.COMPLETED:
            validation_errors = []
            if sample_params.return_token_ids and len(last_token_ids) == 0:
                validation_errors.append("empty response_ids")
            if sample_params.return_logprob and len(last_logprobs) == 0:
                validation_errors.append("missing logprobs")
            if not last_trajectory:
                validation_errors.append("empty response text")
            if self.enable_return_routed_experts and routed_experts is None:
                validation_errors.append("missing routed_experts")

            if validation_errors:
                error_msg = f"Incomplete rollout data for request {uid}: {', '.join(validation_errors)}"
                self.logger.error(f"{error_msg}. Raw response: {response_json}")
                rollout_state.status = Status.FAILED
                rollout_state.error_msg = error_msg
                return rollout_state

        rollout_state.response = last_trajectory
        rollout_state.response_ids = last_token_ids if len(last_token_ids) > 0 else None
        rollout_state.logprobs = last_logprobs if len(last_logprobs) > 0 else None
        rollout_state.routed_experts = routed_experts
        rollout_state.finish_reason = finish_reason
        rollout_state.status = rollout_status

        return rollout_state
