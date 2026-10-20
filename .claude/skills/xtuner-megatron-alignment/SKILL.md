---
name: xtuner-megatron-alignment
description: 对齐 xTuner 与 Megatron 仓库中不同模型的训练精度，发现对应生产入口、验证权重与输入映射，建立同 batch 回放，核对目标函数、优化器、并行与精度语义，诊断数值和性能差异。适用于 dense、MoE、多模态等模型的跨框架训练对比；不将特定模型 recipe 或 RL 链路追踪当作通用流程。
---

# xTuner / Megatron 跨模型训练精度对齐

从两个仓库当前分支最新 commit 开始，针对用户指定模型建立可复现的对齐契约。两边独立完成 forward、backward、梯度累积、裁剪和更新，不注入参考梯度或逐步复制参考权重来制造一致。

## 按任务加载参考

- 建立或接通模型对齐链路：读 [workflow.md](references/workflow.md)。
- 已获授权在 GPU 节点实跑或验收捕获/回放：读 [gpu-runtime-and-tests.md](references/gpu-runtime-and-tests.md)。CPU 侧准备任务仍只交付指令，不因加载参考而自动执行 GPU 操作。
- 判断数学、优化器、并行和dtype是否等价：读 [math-and-parallelism.md](references/math-and-parallelism.md)。
- 分析已运行实验的精度、gradnorm和速度：读 [diagnosis-and-performance.md](references/diagnosis-and-performance.md)。
- 定位两仓库的通用能力和可复用工具边界：读 [repository-map.md](references/repository-map.md)。
- 仅当涉及 Qwen3.5-9B ConceptLM 或明确需要案例时，读 [qwen35-conceptlm-example.md](references/qwen35-conceptlm-example.md)。其中八卡、Muon、VQ/HLM、SP8、数据路径和历史结论不是通用默认值。

## 通用工作原则

1. 先发现两边真正对应的模型及入口，确认结构和数学目标可比。默认仅对比两边已支持的共同模式，不必为了覆盖所有模式新增实现；通过现有配置选择一致模式，并在报告中说明双方如何配置及未覆盖范围。若用户明确指定的模式尚未共同支持，应报告缺口，不擅自换模式，也不默认扩展实现。
2. 使用当前分支最新 commit、用户指定checkpoint和实际运行配置。保留用户已有改动；暂存、修改暂存区和提交只按用户明确授权操作。用户说“不要做git操作”时，默认指不主动暂存、修改暂存区或提交，不应据此关闭或修改训练代码原有的git元数据采集，也不应绕过Trainer。若用户明确禁止所有git调用（包括只读命令和程序内部调用），再遵守该更严格限制。
3. 发现两框架实现差异时，保留原实现并报告差异，在已授权环境中直接启动对比，不先询问是否修改。取得双方可比结果后，再结合实测差异解释影响，询问用户是否修改；得到确认后，正式适配按仓库架构放入config、model/module、loss、optim及checkpoint/data转换层。默认保持其他模型行为；测试或启动脚本不重写forward、backward或优化器数学。
4. xTuner优先使用该任务现有的高层Trainer CLI，通过配置启动Trainer.fit；回放适配接入正式扩展点，不默认自行组装TrainEngine或编写外层更新循环。只有现有Trainer确实无法满足任务、且已说明具体缺口与采用底层入口的理由时，才直接使用TrainEngine。Megatron使用实际生产入口。具体接入方式见 [workflow.md](references/workflow.md)。
5. 严格回放必须同初始状态、同完整输入和同每步调度。相同数据集、tokenizer、seed或.bin/.idx不保证相同batch；独立训练实验与逐步精度对齐分开。
6. 并行拓扑、硬件数量、优化器、loss、dtype和采样方法从当前模型发现，不固定为八卡、SP8、Muon、BF16或某组辅助loss。用户 prompt 未指定 SP 时，默认 SP=1，并在启动脚本、有效配置和结果报告中显式注明；用户明确指定时按其要求配置。此默认不推导或改写 TP、CP、DP 等其他并行维度，也不改变模型原生 group 语义。
7. 遵守执行环境及授权。CPU主机或用户要求自行运行GPU时，只做CPU检查并交付GPU指令；不自动执行CUDA/RJob或队列操作。

## 实现差异处理顺序

两框架的参数分组/weight decay、参数逻辑维度、loss公式及归一化、detach、dtype/cast、离散选码或算子路径等存在差异时，先保留双方原实现。启动前报告已知差异与本次共同模式，然后在已有授权和资源范围内直接运行参考、参考自身rerun与目标框架对比，不把“是否修改实现”作为启动前置确认。

取得双方可比结果后，结合loss、梯度/gradnorm、参数更新等实际差异及归因证据，说明保留原实现的影响、拟修改文件与范围，再明确询问用户是否修改。确认前不写入或启用修正，即使拟采用默认关闭的opt-in开关也一样。没有双方结果时应说明缺失或阻塞，不能用推测代替对比证据并提前要求修改实现。具体报告项见 [math-and-parallelism.md](references/math-and-parallelism.md)。

这一顺序不扩大执行授权：CPU-only任务仍交付原实现对比的启动指令，等待用户提供结果，不自动使用GPU或提交队列。“实现训练对齐”的总任务不等于确认修改所有实现差异。已有明确确认沿用其具体范围，不重复询问；新发现的差异仍先对比、后询问。选择双方已有共同模式仍按前述原则配置并报告，不因此新增其他模式或改写实现。

## 启动前报告与观测要求

启动任何捕获、rerun、回放或独立训练前，先向用户报告两仓库的改动范围：分别说明是否只改配置/训练脚本，是否涉及model/module、loss、optim或engine实现，并列出文件、目的及数学/默认行为影响。明确哪些输入、调度或状态来自记录回放，哪些由当前框架独立计算；无改动或无回放也要说明。CPU侧交付启动指令时一并提供该报告；对已授权的原实现对比直接启动，不先询问是否修改实现；修改意向在取得双方对比结果后再确认。报告还须列明本次选择的共同支持模式、双方配置文件与具体CLI参数/环境变量及取值、checkpoint与该模式的兼容性，并限定结论仅适用于本次模式。

Megatron与xTuner都须在训练log中逐update打印耗时、TPS与TPS/GPU，并将训练及性能指标写入独立TensorBoard runs供同屏比较。双方都接入可配置profiling，默认关闭。字段、计时/吞吐口径、TensorBoard展示和profiling验收见 [diagnosis-and-performance.md](references/diagnosis-and-performance.md)。

每次新增或更新对比实验，都须提供放在 xTuner 仓库内的可复用启动脚本，在脚本中显式列出本次 xTuner 配置及开关值，并接通实际 Trainer 配置读取；已有入口优先更新复用。不能只在对话中列值、依赖隐藏默认值，或将正式脚本仅放在仓库外。参数清单、覆盖规则与验证流程见 [GPU 运行参考的 xTuner 对比启动脚本要求](references/gpu-runtime-and-tests.md)。CPU-only 任务也交付该脚本，但不因此启动 GPU。

## 对齐契约

记录两边：

- 模型变体、训练任务、对应入口和实际执行源码/环境；checkpoint、参数/状态映射和初始化方式。
- 全部真实输入、标签、边界、mask、采样与随机性；动态shape和模态预处理结果。
- 主loss及该模型启用的辅助目标，逐项公式、权重、分母、detach和归约方式。
- 实际optimizer、参数组、LR/schedule、decay、clip、state dtype；无需Muon的模型不引入Muon适配。
- DP/TP/PP/CP/SP/EP与FSDP/HSDP等group语义、每步样本量和累积次数；副本如何去重。
- 参数/计算/master/梯度/state与中间算子的精度、融合、compile/recompute和真实软件版本。

先验证输入与状态，再少量step，最后多步轨迹。先通过完成记录、唯一输入指纹、有效targets、每步实际LR及skipped/nonfinite检查，再解释数值误差。没有本次容差就报告差异，不宣称parity通过。

## 证据与交付

CPU验证文件/权重契约、布局转换、采样与计数、独立loss/optimizer公式、CLI与默认兼容。真实内核、跨卡gather/backward、完整模型更新和吞吐需要目标GPU环境验证，不能由CPU测试替代。

保留参考生产、生产自身rerun和目标框架的独立结果。报告各loss、裁剪前gradnorm、必要的参数/梯度误差、LR、样本/target数及指纹，区分数学、配置、内核舍入和未验证假设。

每次向用户交付对比结果时，在最终回复中同时给出本次输出根目录、对比报告/原始指标保存位置、TensorBoard 同屏对比目录的绝对路径，以及可直接使用的查看命令和执行主机。说明目录中的各 run 对应哪个框架/配置，按实际 event 文件确认路径；不能只报告误差或让用户自行查找。结果路径与已完成实验绑定，不能把后来修改的脚本配置当作历史运行配置。详细要求见 [GPU 运行参考的完成验收与交付](references/gpu-runtime-and-tests.md)。

提交只包含用户指定的正式适配与必要入口；定位测试、历史迁移、调度脚本分开。固定GBS/序列长的文本实验可用 scripts/summarize_performance.py，只读汇总相同步骤；多模态或动态token量应按真实工作量扩展口径。
