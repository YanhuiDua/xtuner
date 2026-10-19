# GPU 侧运行与测试

本参考补充已获授权的 GPU 实跑编排、环境与资源检查、捕获/回放和完成验收，不包含精度问题定位。数学适配与输入契约见 [workflow.md](workflow.md)。CPU 侧开发、仅要求给命令或用户自行执行的任务，不自动升级为 CUDA 或 RJob 操作。

## 1. 确认节点、共享存储与当前任务

- SSH alias 是入口，不是节点身份。连接后核对 hostname、GPU 型号/数量、仓库和 checkpoint/data 挂载。节点重建、排队重新分配后 IP、host key、容器进程空间及环境均可能变化。
- 主机密钥变化时先核实身份；不关闭检查、不自动删除 known_hosts。用户要求暂不连接时，不再尝试该节点，也不把以前确认过的另一个指纹当成本次授权。
- 用户给出排队作业时，可按授权通过 CPU 管理节点查询 `rjob get "$JOB_NAME" --namespace="$NAMESPACE"`。获得有效分配且核实归属后才更新目标 alias 的 HostName，保留其余 SSH 配置。作业不存在/未运行、查询环境不可用或身份未核实时，不猜测 IP 或自行重建作业。
- RJob 客户端可能依赖管理节点的标准初始化环境；按节点既有方式加载。不要打印整个环境或 `/proc/*/environ` 来排错，尤其不要把凭证写入日志。
- 共享文件可从可用节点读取或编辑，不必为了修改脚本连接暂不可用节点。共享路径存在不代表在所有节点可写；核对 UID、目录权限和空间，在用户授权范围内写入。

运行前检查 GPU 和训练进程：

```bash
hostname
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
ps -eo pid,ppid,args
```

按已知入口、run-dir、PID/PGID 确认进程归属，输出按需过滤。`nvidia-smi` 的 PID 可能来自宿主机命名空间；`[Not Found]` 不等于没有任务。GPU 利用率瞬时为 0 也不代表空闲。已有其他训练时保留其进程和运行状态，先做 CPU 检查；只有取得相应授权才能中断它。

## 2. 锁定实际 Python 与依赖

运行前同时读取当前 launcher、内层 shell 和最终 Python 入口。有脚本显式重置 PATH 或调用 `/usr/bin/python`，所以激活 venv/Conda、`which python`、脚本名称都不足以证明实际解释器。

用最终被调用的解释器记录：

- `sys.executable`、Python、Torch/CUDA、Triton、Transformer Engine、FlashAttention/FA3、FLA、causal-conv1d 等模型实际使用的版本。
- 导入模块的 `__file__`；distribution 版本与实际 import 来源都要检查。venv 可能只覆盖某一个包并继承系统其他包。
- 启用的 backend、compile/recompute、sharding、no_sync 等有效配置，以及环境变量的允许列表；不要保存整个 shell 环境。

使用启动链末端实际可控制的解释器或环境，不只改外层 PATH。首次短跑日志应能确认环境选择。发现使用了错误环境时，该记录保留并明确标记，重新在指定环境测试；不能沿用错误环境的结果作为目标实验结论。

编译缓存放当前用户可写位置，记录 cache 路径及冷/热缓存。复用时核对框架、Python、CUDA/Triton/TE 与源码兼容；不要默认依赖另一位用户的预编译归档。编译耗时不算稳定训练吞吐。

## 3. 在仓库内维护入口，结果独立存放

可复用启动、转换、回放、验证脚本放相应仓库目录，确保 Git 可发现。临时传输可经过 `/tmp`，最终入口不要只存在于仓库外。结果、日志、TensorBoard、runtime staging 和编译缓存可放外部可写目录。

只是重跑或改结果目录时，优先复用现有入口及其 output/run-dir 参数，不另造一个 launcher。入口变动后，重新读取当前 CLI 与失败日志，提供匹配当前版本的命令；用户要求只更新命令时不顺带改代码。

每次运行创建新目录，保留原捕获、Megatron rerun 和各 xTuner 配置记录。名称至少能区分模型、拓扑/关键配置与时间或唯一 ID。时间标注时区，避免节点 UTC 与用户本地时间混淆。不要只换 metrics 文件名而让 completion、config、TensorBoard 或比较输出覆盖旧结果。

原捕获只读。旧捕获与新入口仅封装/schema 不兼容时，在新目录转换，记录来源、转换前后指纹及字段变化；保留真实 tokens、labels、mask、边界、LR 和权重来源。模型/optimizer 契约不兼容不能靠改 metadata 宣称兼容。检查转换工具是否写死步数、GPU 数、GBS 或参数量。

运行前保存实际命令、commit、相关工作区 patch、executed launcher/入口、checkpoint/config/输入契约和环境版本。保留已有用户修改，不自动 checkout/reset/pull，也不把提交源码等同于实际工作区源码。

### xTuner 对比启动脚本（每次对比必备）

每次对比须创建或复用 xTuner 仓库内的正式启动脚本，例如放在该模型已有的 `examples/` 或训练脚本目录。配置参数集中在脚本的可读配置区，调用该任务现有 Trainer CLI；配置文件负责把这些值接入模型、FSDP/engine 和 profiler。仓库外的旧入口如需保留，仅作兼容转发，不另存一套默认值。不通过执行 git 命令来证明文件位于仓库内。

脚本须显式体现以下适用项；不支持或不适用时写明原因，不虚构开关：

| 类别 | 显式记录内容 |
|---|---|
| 输入与规模 | checkpoint/捕获来源、独立输出目录、GPU 数/并行拓扑、SP、GBS、MBS、累积次数、序列长度和步数；回放固定值可以从捕获读取，但须打印实际值及来源 |
| Attention | attention backend 与 FA3 开关分别列出；FA3 开关和 backend 的组合须真正启用对应实现，不能仅设置环境变量 |
| 分片与累积 | ZeRO/FSDP 的实际选项、`reshard_after_forward`、`gradient_accumulation_no_sync`；说明 ZeRO 命名到 FSDP2 参数驻留的真实映射，单卡不宣称验证了跨卡分片 |
| 编译与重计算 | `compile_cfg`/实际编译开关、`recompute_ratio`，以及会影响本次模型的其他重计算设置 |
| 精度 | 参数计算与梯度归约 dtype，以及本次使用的混合精度设置 |
| Trace | 独立的 trace/profiling 开关、start/warmup/active 等实际支持的窗口、step 单位和输出位置；默认关闭。普通耗时/TPS 日志不等于开启 trace |

取值来自用户选择及本次实验配置，不把某次 FA3、compile、ZeRO 或重计算取值固化为其他模型的默认值。允许用环境变量/CLI 覆盖时，在脚本显式写出默认值与覆盖方式；读取到的值必须传入实际配置，不能只 export 一个下游未使用的变量。互斥或不支持的组合应明确报错，不静默回退。仅为显式化配置，不改模型、loss 或 optimizer 数学。

交付前检查 shell/Python 语法，并通过只加载配置、不构造训练引擎或调用 `fit` 的方式核验默认值及代表性覆盖值确实传到配置。GPU 实跑仍按授权进行，并从首轮日志/有效配置核验真实 backend、分片、no_sync、compile、重计算与 trace 状态；配置加载通过不代表 CUDA 功能已验收。启动时打印这些有效值，保存本次脚本及有效配置快照，运行期间不原地改写正在执行的 shell 脚本。交付仓库内脚本路径、实际命令和结果目录，而非仅在回复中罗列参数。

## 4. 捕获、参考 rerun 与目标回放

先短跑 smoke，核实加载、数据、backward、clip、optimizer、completion 和退出处理，再执行用户指定步数。完整多步测试要包含真实连续更新；只做无更新观测、forward-only 或跳过 optimizer 的测试不能称完整训练回放。

| 模式 | 输入与入口 | 保存内容 |
|---|---|---|
| Megatron 捕获 | 模型真实生产 loader/入口，附加只读捕获 | 完整 batch、调度、完成记录、metrics、状态/来源 |
| Megatron rerun | 同初始 checkpoint，用生产回放入口读取成功捕获 | 独立 `megatron_rerun` 结果，不覆盖参考捕获 |
| xTuner 回放 | 同初始权重，读取对应捕获，经标准 engine 更新 | 独立配置/完成记录/metrics/TensorBoard/比较结果 |

修改硬件数量时，显式保留用户要求的生产默认值和例外项，不把旧实验的默认值带入新入口。GBS、MBS、真实 DP size 和累积次数一起核对；保持 GBS 不等于保持每卡 micro 数。短测步数与完整生产 schedule 是独立配置，不应因为只跑 100 步就把 warmup/decay 也自动缩为 100 步。

启动后台任务时记录 runner/训练 PID、run-dir、日志和退出码。使用现有稳定 runner；远端交互 shell 断开不应杀死已授权的训练。`nohup/setsid` 返回不等于启动成功，需查看初始日志及实际 worker。使用 `tee` 的 shell 管线要保留训练失败码，不能把 tee 的成功当训练成功。

## 5. 完成验收与交付

先验收原始产物，再运行比较或性能汇总：

- 各必要 rank/stage 的 completion、runner/training 退出码、实际完成 step 范围；检查重复、缺失、skipped 和 nonfinite。仅 loss-stage 输出 metrics 的模型按真实拓扑验收。
- micro 数、数据身份/指纹、有效 targets、实际 LR、源捕获完整性；不要只数目录或 TensorBoard event 文件。
- 将训练与比较/上传的完成情况分开记录。训练完成但后处理失败时，优先仅重跑后处理，不重复昂贵训练或覆盖结果。
- 收到“已跑完，确认状态”时先核对这些证据和当前进程，不自动重跑。统计完整记录只计通过契约的多步结果，区分原生捕获、rerun、回放、失败/中断和短 smoke。
- TensorBoard 路径从实际 event 文件和 launcher 配置确认，给出本次独立目录；symlink/live 目录可能指向后来运行，需说明目标。原始 JSONL 是指标验收依据。
- 性能只比较共同完成步骤和相同工作量，排除编译启动与 profiler 窗口，并披露计时边界，见 [diagnosis-and-performance.md](diagnosis-and-performance.md) 的性能部分。此处不包含精度定位流程。

交付实际运行命令、输出目录、训练/后处理完成情况、有效配置与环境及验证范围。

## 6. 本对话的 Qwen3.5 操作实例

这些是历史节点/工具约定，使用前重新核对，不适用于其他模型默认配置。详细模型路径与入口见 [qwen35-conceptlm-example.md](qwen35-conceptlm-example.md)。

- 曾使用 `ssh sf_gpu_mg` 跑 Megatron、`ssh sf_gpu_xt` 跑 xTuner，两边共享 `/mnt/shared-storage-user/llmrazor-share/heqiushi/`。alias、IP 和可用资源会变化。
- Megatron `examples/conceptlm/framework_alignment/run_megatron.sh` 用新的 `ALIGNMENT_RUN_DIR` 捕获；rerun 还需 `MEGATRON_REPLAY_SOURCE` 指向源捕获、`PRETRAIN_ENTRYPOINT` 指向同目录 `production_replay.py`。读取当前 preflight/CLI 后再组命令。
- xTuner `examples/v1/scripts/megatron_alignment/run_xtuner.sh` 用 `ALIGNMENT_RUN_DIR` 传给 Python 入口的 `--run-dir`，读取已准备的回放目录；同目录 `run_compile_recompute0.sh` 的第一个位置参数是 output 目录，还包含后处理。不能把两者参数或副作用混用。
- 历史 GBS128、MBS1、DP8、100 步捕获，每 rank 每步 16 个 micro，共 12800 个独立 batch；历史 GBS8 对应 800 个。使用其他拓扑或步数时重新计算，不能把这些数字写成通用验证常量。
- 某些入口显式使用 `/usr/bin/python`，另有只覆盖 FLA 的 `venv-fla051`。按本次版本要求选择并验证；不是看到 venv 名就认定 Torch/TE/FLA 全部符合要求。

Windows PowerShell 向远端提交多行 Bash 时，曾使用 UTF-8/Base64 避免 CRLF 和多层 shell 引号问题。下面只演示只读命令，实际脚本仍放仓库：

```powershell
$remoteScript = @'
set -euo pipefail
hostname
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript.Replace("`r", "")))
ssh sf_gpu_xt "printf '%s' '$encoded' | base64 -d | bash"
```

不要把 PowerShell/JavaScript 的字符串编码当 Bash 转义。带 checkpoint 路径、美元符号或反引号的命令需检查每一层的实际传递语义。
