import os
from copy import deepcopy

from transformers import AutoTokenizer
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets import RLTextTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model.dense.qwen2 import Qwen2Dense7BConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
from xtuner.v1.rl.base import WorkerConfig

from ppo_project.train.ppo_trainer import RLTrainerConfig
from ppo_project.rl.ppo import CriticLossConfig, GRPOLossConfig
from ppo_project.rl.base.worker import PPOWorkerConfig
from ppo_project.model.qwen25dense_rm import Qwen25DenseRMConfig
from ppo_project.model.qwen2dense import Qwen25Dense05BConfig


work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_evaluate = True if eval_data_path != "" else False

# basic settings
experimental_name = "ppo_math"
total_epochs = 1000
global_batch_size = 1024
prompt_repeat_k = 1
rollout_tp_size = 2
rollout_ep_size = 1
max_prompt_length = 1024
max_response_length = 512
pack_max_length = 16384
train_optimizer_steps = 4
hf_interval = 50
enable_evaluate = True
enable_initial_evaluate = True
evaluate_step = 5

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.8,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=0,
    min_tokens=0,
)

# dataset
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
eval_dataset = DatasetConfig(name=experimental_name, anno_path=eval_data_path) if enable_evaluate else None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)

train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}] if enable_evaluate else []

dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")

# 3. judger
from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
judger_cfg = JudgerConfig(reward_judger_configs=[gsm8k_judger_config])

# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
)


evaluator_cfg = EvaluatorConfig(
    enable_evaluate=enable_evaluate,
    enable_initial_evaluate=enable_initial_evaluate,
    dataset_cfg=eval_dataset_cfg,
    tokenizer=tokenizer,
    evaluate_step=evaluate_step,
    sample_params=evaluation_sample_params,
) if enable_evaluate else None

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg, dataloader_cfg=dataloader_config, tokenizer=tokenizer
)

# 5. Train worker
train_worker_cfg: WorkerConfig = PPOWorkerConfig(
    # policy_model_cfg=Qwen25DenseConfig.from_hf(args.policy_model_path),
    policy_model_cfg=Qwen25Dense05BConfig(),
    policy_optim_cfg=AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False),
    policy_loss_cfg=GRPOLossConfig(
        policy_loss_cfg=dict(
            cliprange_high=0.2,
            cliprange_low=0.2,
            loss_type="vanilla",
        ),
        ignore_idx=-100,
        use_kl_loss=False,
        kl_loss_coef=0.001,
        kl_loss_type="low_var_kl",
        mode="chunk",
        chunk_size=512,
        gamma=1.0,
        gae_lambda=1,
    ),
    policy_load_from=model_path,
    critic_model_cfg=Qwen25DenseRMConfig.from_hf(model_path),
    critic_optim_cfg=AdamWConfig(lr=1e-5, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False),
    critic_loss_cfg=CriticLossConfig(
        critic_loss_cfg=dict(
            clip_epsilon=0.5,
            loss_type="ppo",
        ),
        ignore_idx=-100,
        mode="chunk",
        chunk_size=512,
        gamma=1.0,
        gae_lambda=1,  # 0.95
    ),
    critic_load_from=model_path,
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
    fsdp_cfg=FSDPConfig(
        torch_compile=False,
        cpu_offload=False,
        ep_size=1,
    ),
    sp_size=1,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 6. RL Trainer
trainer = RLTrainerConfig(
    load_from=model_path,
    resources=resources,
    rollout_config=rollout_config,
    dataflow_config=dataflow_config,
    judger_config=judger_cfg,
    replay_buffer_config=replay_buffer_cfg,
    evaluator_config=evaluator_cfg,
    train_worker_config=train_worker_cfg,
    tokenizer_path=model_path,
    work_dir=work_dir,
    total_epochs=total_epochs,
    hf_interval=hf_interval,
)
