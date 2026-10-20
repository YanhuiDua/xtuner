# 案例：Qwen3.5-9B ConceptLM

仅在任务明确涉及该模型或需要参考落地案例时读取。以下公式、参数、脚本、八卡拓扑和历史观察均不适用于其他模型的默认流程。执行前核对当前源码；案例不是通用配置模板。

# 模型专用数学与 SP 实现

## ConceptLM 目标

本会话默认 chunk4、VQ/HLM 权重1：

```text
T = 跨全部 microbatch、唯一 DP 样本的有效 target token 数
L = [CE_num + 4*(VQ_num + HLM_num)] / T
```

- VQ 输入在 RMSNorm 前 detach；FP32 L2 distance、argmin 和 codebook lookup；hidden 维平均 MSE，commitment=0.25。codebook 有梯度，encoder 不从 VQ 输入获得梯度。
- concept valid mask 为 chunk 内任一 token 有效，不是要求全部有效。
- HLM 用 normalized predicted 预测同文档的下一个 normalized concept；target detach，排除文档末尾和无效 successor。SP 切分点不是文档末尾。
- 复用 concept_quantizer 的无状态函数，保持原有 codebook 参数名字和数量；HLM 复用 next_concept_mse 和原序列边界能力。调用方负责 norm/detach。
- CE token reduction 与 square/sample reduction 不是同一目标；mask、shift 和分母必须对齐。

## Muon 参数更新

本会话 opt-in MuonConfig(megatron_compatible=True)，默认 False 保留普通实现。

| 项目 | Recipe |
|---|---|
| momentum | EMA m=mu*m+(1-mu)*g，mu=0.95 |
| Nesterov | g.lerp(m, mu) |
| NS | FP32 norm，eps1e-7，BF16 五步 quintic，FP32 输出 |
| 逻辑矩阵缩放 | 0.2*sqrt(max(rows,cols)) |
| AdamW fallback | betas=(0.9,0.95)，eps1e-8 |
| 默认 decay | 0.1；norm/1D/路由系数等按实际参数组覆盖 |
| clip | 全参数阈值1.0，记录裁剪前 norm |
| LR | 每步实际值，含首步0 |

按语义矩阵而非仅按物理 ndim 分组：

- MLA Q-up 每完整 head=192+64行；KV-up 每head独立K192/V256；KV-down512/64。
- GDN fused in-proj 的 Q/K/V 逐head Muon，尾部 z/beta/alpha Adam。
- SwiGLU gate/up 分开；MUDD router-w2 按独立角色拆分。
- 路由二维 norm/static coefficients 走 Adam/no-WD；conv 的3D/2D、fusion scalar/[1]布局差异不应改变算法或 decay。
- 若主更新已经对整个 fused 参数 decay，Adam 尾部不能再次 decay。

专用 MegatronMuon 复用原 DTensor 通信/分片能力和标准 AdamW，隔离 EMA/NS。不要改普通 Muon 的动量、bias correction 或默认 NS 算式来迁就特定模型。fresh optimizer 测试不证明跨拓扑 checkpoint state 迁移正确。

## SP 归一化

CE context 对 labels 做 SP split，分母统计唯一有效 token，复用可微 WORLD SUM 和 FSDP 平均。

当前 auxiliary 路径让每个 SP peer 使用完整 gathered concept/predicted 和完整 mask：SP size=S 时，重复分子和分母相抵 S*N/(S*T)=N/T。不能只验证 forward；backward 还经过 WORLD autograd reduction、sequence gather backward 和 FSDP averaging。测试不同 DP/SP、不同有效长度，检查 codebook 和预测路径梯度。

接入条件：

1. 创建 (world/SP,SP) 数据 mesh，native sampler 用 DP-local rank，SP peers 获得相同完整样本并校验指纹。
2. SequenceContext.split(sp_mesh) 切 token；完整 labels 由 native CE context 切分。
3. VQ/HLM 保留完整 mask/document boundaries，用 full gathered features，正确处理跨 SP successor。
4. 每卡 microbatch 数=GBS/DP_size，不固定为GBS/world。
5. 日志 targets、auxiliary numerator 去除 SP 副本，回放指纹保留原始 source rank 归属。
6. FSDP 参数分片可仍覆盖全world，不等于只在 DP 子组分片。

8192长度、chunk4、SP8：每卡输入1024token、局部256concept、完整2048concept。GBS128 下 SP1/DP8 累积16次，SP8/DP1累积128次。此 auxiliary 方案重复完整序列计算，属于复用现有能力的正确性方案，并非 auxiliary 完全分片优化。

## dtype 和数值内核

数学等价、配置一致、容差内一致、bitwise相同是不同结论。

- 显式核对计算、master、state、归约 dtype。FSDP reduce 默认可能是BF16，不能假设与 Megatron BF16+FP32 main grads 一致。
- FP32 保存 pattern 是否还意味着参数被 FSDP 忽略并 FP32 运行，要读当前 sharding 实现。
- FSDP 梯度生成后 cast 与 TE FP32 wgrad accumulate 的累积顺序不同。
- TE CE vs chunk CE：投影分块、logits cast、有效 vocabulary/padded rows、backward 归约。
- NS：归一化前 cast、逐矩阵/批矩阵、addmm融合、Triton SYRK、系数与迭代次数。CPU 普通 EO 对齐不证明生产 SYRK 对齐。
- GDN：A_log.exp 与 float().exp、Q/K L2norm 在 head repeat 前/后、norm-gate backward。compile 会融合中间 cast，不能仅据源码推断实际 GPU 舍入。
- compile=True 不等于所有算子 fullgraph；核对 target、graph break。固定 dropout/RNG、recompute RNG preservation、FA/TE/FLA/Torch/Triton 版本和 matmul precision。

## 可选 LM z-loss

本会话只讨论过方案，不应默认称已启用。先读当前入口：

```text
Z_t=lambda*logsumexp(raw_LM_logits over valid vocabulary)^2
L=[sum(mask*(CE+Z))+4*(VQ_num+HLM_num)]/T
lambda=1e-5 是本会话方案，非所有任务的固定要求
```

Megatron 复用 examples/conceptlm/lm_zloss.py；xTuner 放正式 CE/loss context，复用 chunk logits。FP32 logsumexp/square、mask、词表、分母和梯度一致；分别记录纯CE、z-loss、VQ、HLM、total。

不要直接换 pretrain_with_streaming_zloss.py：历史专用入口还绑定 InternLM、60k resume、raw_logits、affine norm 恢复及 embedding/head no-WD，会同时改变其他对比条件。

# 专用脚本、数据与历史实例

从本skill目录到仓库根是 ../../..。既有 .agents/skills 链接到 .claude/skills，不创建重复技能目录。

## 本会话路径：每次使用前重新核对

- xTuner：/mnt/shared-storage-user/llmrazor-share/heqiushi/ncp/qwen35/xtuner-ncp
- Megatron：同级 megatron_ncp_qwen35，从当前分支最新 commit 开始。
- 拷贝raw checkpoint：同级 megatron_checkpoint_iter0000002。
- HF conversion：从本次 checkpoint 转换产物及其 manifest 确认路径，核对 config、参数映射和张量内容。
- Dolma：/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025。
- 原始文本 data/ 当时有6081个.jsonl.zst；tokenized recipe是megatron_qwen35_8192/，data_args_path.txt当时引用16组.bin/.idx；tokenizer在其tokenizer/Qwen3.5-0.8B-Base。

挂载确认后继续检查真实文件，不重复请求确认。外部环境、compiler cache和数据可只读；只把实际读取的外部路径列为依赖，不要求挂载历史用户的全部目录。

## xTuner 相对仓库根

默认脚本目录 examples/v1/scripts/megatron_alignment/：

| 文件 | 职责 |
|---|---|
| replay_xtuner.py / run_xtuner.sh | 生产batch/LR回放，八卡标准TrainEngine，SP1/SP8 |
| batch_contract.py | 元数据、dtype/shape/内容指纹和mask/边界检查 |
| compare_results.py | completion、step/micro、targets、指纹、LR和误差报告 |
| convert_capture_format.py / compare_converted_capture.py | 历史封装转换，先检查固定参数和证据适用范围 |
| train_qwen35.py / xtuner_text_source.py / qwen35_wsd_lr.py | 独立native indexed/text数据和本地LR，无参考batch/LR依赖 |
| ../run_qwen35_8gpu.sh | 独立八卡入口，默认SP1，SP_SIZE=8或--sp-size 8 |
| probe_xtuner_gradients.py / locate_xtuner_gradients.py / gradient_probe_utils.py / layer_gradient_trace.py / compare_gradient_location.py | 诊断工具，与正式训练分离 |

正式能力：

- xtuner/v1/loss/conceptlm_loss.py：VQ/HLM context、token校准、SP full-feature路径。
- xtuner/v1/model/dense/conceptlm_qwen35_v2.py：模型、loss接入、Muon语义布局、SP gather。
- xtuner/v1/module/concept_quantizer.py / concept_sequence.py：共享量化和HLM序列能力。
- xtuner/v1/config/optim.py、xtuner/v1/optim/megatron_muon.py / muon.py：opt-in适配、共享通信和原有默认实现。
- examples/v1/scripts/align_conceptlm_qwen35_v2_sp_training.py：原SP测试；随机数据、AdamW/CE默认配置不等于生产对比recipe。

run_gradient_*、wait_then_run_nosync.sh、run_qwen35_controlled.sh、run_compile_recompute0.sh可能等待GPU、绑定固定路径；先读代码，存在不等于获准执行。

tests/optim/run_megatron_muon_cpu.py、tests/utils/_cpu_source_loader.py、test_megatron_alignment.py是低层回归，不在正式回放依赖链。source loader可能读取git show HEAD；用户仅限制暂存、修改暂存区或提交时，不应据此禁用该原有只读行为；用户明确禁止所有git调用时不能直接运行这条测试路径；Gloo测试reshape shim不应进入生产实现。

## Megatron 相对其仓库根

- examples/conceptlm/submit_conceptlm_qwen35_9b_block4_8k_fastest.sh：生产配置、融合、优化器参数。
- examples/conceptlm/train_conceptlm_qwen35_2b.sh：通用launcher，名称2B不等于实际config不是9B。
- examples/conceptlm/pretrain_conceptlm.py：真实模型/数据/loss入口。
- examples/conceptlm/framework_alignment/run_megatron.sh：捕获启动和可写runtime。
- 同目录production_capture.py / production_replay.py：捕获和生产rerun；preflight.py、batch_contract.py、data_contract.json是契约检查；compare_megatron_replay.py比较自身漂移。
- examples/conceptlm/lm_zloss.py：可复用z-loss。
- core/optimizer/muon.py可能只是shim，要追踪运行环境实际emerging_optimizers版本和实现。

## 历史结果目录

根目录 /mnt/shared-storage-user/llmrazor-share/heqiushi/megatron_compare_runs：

- GBS8：aligned_conceptlm9b_sp1_100_20261015_174834；xTuner子目录xtuner_gbs8_compile_recompute0_20261016_135944。
- GBS128：aligned_conceptlm9b_sp1_gbs128_100_20261016_134711；子目录megatron_rerun及xtuner_gbs128_compile_recompute0_20261016_142849。

以本次文件为准。后续诊断子目录可能包含新证据，不能无条件沿用历史假设。

# 原会话的诊断与性能实例

GBS128 的部分尖峰样本是8192-token的 `000,` 周期，由6144个token15和2048个token11组成，几乎全部参与loss；另有 `1,0x...`。step27有同类周期样本但两边norm接近，因此是触发关联而非充分根因。

step21裁剪前norm约137.21（Megatron）、102.66（rerun）、370.33（xTuner）；step0约159.23/159.25/208.73，而total loss差约0.019%。当时优先怀疑数据相关的反向数值敏感性，未锁定单个算子；不能将它作为其他模型尖峰的根因。

8xH200、seq8192、100步，排除前10步：

| 配置 | xTuner平均秒/步 | Megatron平均秒/步 | 比例 |
|---|---:|---:|---:|
| GBS8，compile/recompute0 vs原跑 | 1.583 | 1.274 | 1.243 |
| GBS128，compile/recompute0 vs rerun | 21.454 | 11.840 | 1.812 |

当时GBS128 xTuner no_sync=False、reshard=True，不能套用于后来的参数驻留/no_sync入口。两个GBS实验的LR与计时边界不同，不能由性能数字推断精度或量化通信贡献。

专用 GPU 命令示例，路径及环境由任务设置：

```bash
export MEGATRON_ROOT=/path/to/current/megatron
export XTUNER_ROOT=/path/to/current/xtuner
export ALIGNMENT_RUN_DIR=/path/to/new/capture
export LOAD_DIR=/path/to/copied/raw_checkpoint
export HF_PATH=/path/to/matching/hf_conversion
export DATA_ROOT=/path/to/indexed_recipe
TRAIN_ITERS=5 GLOBAL_BATCH_SIZE=8 \
  bash "$MEGATRON_ROOT/examples/conceptlm/framework_alignment/run_megatron.sh"
# 切换到xTuner GPU环境，继承捕获目录。
bash "$XTUNER_ROOT/examples/v1/scripts/megatron_alignment/run_xtuner.sh" \
  --sp-size 1 --compile --recompute-ratio 0
python "$XTUNER_ROOT/examples/v1/scripts/megatron_alignment/compare_results.py" \
  "$ALIGNMENT_RUN_DIR" --sp-size 1
```

SP8回放传 --sp-size 8；独立训练用 SP_SIZE=8 bash examples/v1/scripts/run_qwen35_8gpu.sh。回放脚本和比较工具存在同名completion/config输出，配置不同应使用独立结果目录。历史转换工具可能固定100步/800batch/960参数，不应直接用于其他模型。
