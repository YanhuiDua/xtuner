# 发现两仓库的模型能力

从本skill目录到xTuner根是 ../../..。仓库通过 .agents/skills 链接到 .claude/skills，沿现有组织放置能力，不创建重复技能目录。Megatron根路径由用户任务或环境发现，不固定用户目录。

## xTuner 通用发现入口

先读取用户提供launcher和config，再用以下位置查找对应实现；不是固定某个训练CLI：

| 位置 | 应查内容 |
|---|---|
| examples/、recipe/ | 该模型/任务真实配置、数据和启动方式 |
| xtuner/v1/train/cli/、xtuner/v1/train/trainer.py | 任务CLI、Trainer初始化/循环、loss context、scheduler、profile和保存 |
| xtuner/v1/engine/ | 标准engine与训练更新接口 |
| xtuner/v1/model/dense/、moe/、compose/ | dense、专家、多模态组合模型及其配置 |
| xtuner/v1/model/base.py、model/utils/ | HF/参数映射、sharding、输入和loss context接口 |
| xtuner/v1/module/、ops/ | 模型原生算子、融合与autograd |
| xtuner/v1/loss/ | CE、MoE、MTP、其他实际目标的config/context与全局校准 |
| xtuner/v1/config/optim.py、optim/ | 实际optimizer/group构建和内核 |
| xtuner/v1/datasets/、data_proto/ | 数据集、tokenization/模态预处理、sampler、collator、sequence/context |
| tests/、autotest/ | 该模型/接口已有回归与分布式测试 |

优先从任务现有Trainer CLI和配置接入，入口选择与直接使用TrainEngine的例外条件见 [workflow.md](workflow.md) 的“目标框架回放与正式接入”。发现engine接口不代表应绕过Trainer。独立测试脚本的默认模型/optimizer/loss不一定是生产recipe，必须逐项核对。

## Megatron 通用发现入口

从其当前分支最新commit和用户实际launcher开始。仓库根 pretrain_gpt.py、pretrain_hybrid.py、pretrain_mamba.py、pretrain_vlm.py只是可能入口，模型扩展也可能在examples/下；以实际代码为准。

| 位置 | 应查内容 |
|---|---|
| examples/、用户launcher | 模型变体、模块spec、全部训练参数及data recipe |
| megatron/training/ | 初始化、生产训练循环、data/scheduler/checkpoint |
| megatron/core/models/、transformer/、ssm/ | 对应模型、层、专家/attention/状态空间模块 |
| megatron/core/pipeline_parallel/、distributed/ | microbatch调度、grad finalize、并行归约和分片 |
| megatron/core/optimizer/ | 参数分组、主权重/state和更新；可能委托外部库 |
| 模型data/tokenizer/conversion工具 | 输入与权重映射，按实际入口追踪 |

外部kernel或optimizer依赖读取运行环境实际版本，不能仅检查仓库中的shim。

## 如何复用已有对齐工具

本仓库 examples/v1/scripts/megatron_alignment/ 含真实捕获回放、比较、诊断、独立训练等工具，但其训练入口绑定Qwen3.5-9B ConceptLM配置。新模型只能复用适用的数据指纹、完成检查、观测/比较方法；不要直接调用绑定该模型的TrainEngine配置。

先检查：字段schema、rank/micro身份、loss定义、canonical参数映射、完成marker与输出是否同名、CLI是否写死world/steps/vocab、调用链是否有调度副作用。适用时复用；不适用时按模型架构扩展正式接口和必要编排，不复制整套数学实现。

诊断/CPU source loader可能绕过GPU包初始化或读取git元数据；执行前检查副作用和用户限制。等待队列、GPU占用或RJob工具不是普通比较入口，不自动运行。

Qwen专用文件、绝对路径和历史结果统一收录于 [qwen35-conceptlm-example.md](qwen35-conceptlm-example.md)，只在该模型任务或案例复用时读取。
