# 模型无关的对齐流程

## 1. 发现对应模型与生产入口

从当前分支最新commit和用户给定入口追踪配置、模型类、模块、loss、数据源、engine和optimizer。读取执行目录的executed script/source patch、launch/config及环境版本，不能只看当前文件或入口名称。

建立模型能力表：dense/MoE、attention或SSM、模态、主干/任务头、MTP或其他辅助分支、共享参数、训练阶段。确认两边对应的是同一变体、相同层数/维度/激活/norm/位置编码/目标。相似名称不代表数学等价。

默认从两边现有能力的交集中选择本次对比模式，使用已有配置接通；不必为了对齐所有模式、匹配另一边默认值或覆盖更多变体而新增模块实现。某模式只有一边支持时，列出差异与未覆盖范围即可；仅在用户明确要求该模式或扩展实现时，按指定范围处理缺口，不擅自改换模式。报告中给出双方配置文件、具体CLI参数/环境变量及取值，并核对初始checkpoint与所选模式兼容；结论只覆盖实际验证的模式。不能静默禁用特性或替换模型结构。

发现双方实现不一致时，按 [math-and-parallelism.md](math-and-parallelism.md) 的“先用原实现运行对比，再确认修改”执行：保留原实现，报告已知差异，在已授权环境中直接运行对比，不先询问是否修改；取得双方可比结果后，再结合实测影响询问是否修改optimizer分组、loss、dtype或选码路径。CPU-only任务交付原实现的对比指令，不自动启动GPU。

## 2. 固定初始权重与训练状态

1. 验证用户checkpoint的config、iteration、分片与格式，找到该模型已有转换/加载接口。
2. 建立canonical参数映射，处理重命名、transpose、fused/split projection、专家顺序、模态模块、tied权重、norm参数化和padding维度。
3. 按实际加载后的dtype/shape/张量内容核对。metadata SHA和shard大小只是完整性检查，不是全权重一致的证明。
4. fresh optimizer对比不恢复历史state；resume还需验证optimizer/state、scheduler、RNG、数据cursor和并行拓扑映射。
5. 对某个step做无更新的forward/backward，再开启独立多步更新；不逐步复制参考权重来掩盖差异。

runtime staging、临时文件、编译缓存和结果放可写路径，不改共享只读生产文件。保留实际环境及有效配置，而非只记录入口默认值。

## 3. 捕获完整输入

使用用户所选模型的真实Megatron生产loader，不替换成随机token或其他模型的数据入口。捕获其实际forward输入和预处理结果：

| 类别 | 可能需要的内容，按实际模型选取 |
|---|---|
| 文本 | tokens/embeddings、已shift的labels、position IDs、attention/loss masks |
| packed/变长 | cu_seqlens、有效长度、padding、文档边界及position reset |
| 图像/视频/音频 | 预处理后的tensor、grid/patch/frame/音频长度、模态mask/索引、额外位置编码 |
| 多任务 | task IDs、额外targets、样本权重、辅助分支输入 |
| 状态/随机性 | recurrent/cache reset语义、dropout/数据增强、随机目标的有效输入或可复现状态 |
| 调度 | update/micro/原始数据身份、真实LR、参数组调度变化、样本数和有效targets |

分清原始样本、预处理输入、tokenized文件和捕获batch。相同seed不足以跨库重现不同数据增强或采样实现；需要直接回放实际输入或验证预处理完全一致。

指纹覆盖字段名、dtype、shape、内容和影响训练的元数据。分布式记录标明数据副本和真实sample/micro身份，不将SP/TP副本当新数据。保存CPU contiguous tensors并原子完成文件。成功完成后才写completion；区分任务提交超时、训练失败和后处理失败。

观测wrapper只调用原函数、读取结果并保留返回语义；任何额外跳步终止行为要披露。不要在捕获中改数学实现。

## 4. 目标框架回放与正式接入

- xTuner首先发现并复用该任务现有的Trainer CLI，以配置文件启动Trainer.fit，由Trainer管理初始化、训练循环、累积、裁剪、optimizer更新、scheduler时点、日志和checkpoint。启动脚本只做参数与配置转交，不自行组装TrainEngine或另写更新循环。
- 捕获输入、初始状态、逐步LR及观测优先接入正式dataloader、checkpoint loader、scheduler、hook或必要的可选Trainer扩展点。用户限制主动暂存、修改暂存区或提交时，保留Trainer原有git元数据采集，不新增禁用开关，也不因此绕过Trainer；仅在用户明确禁止程序内部git调用时处理该冲突。默认保持其他任务行为。
- 只有确认现有Trainer无法满足本次任务时，才采用直接TrainEngine入口；交付时说明具体能力缺口、已评估的高层接入方式，以及底层入口与生产Trainer的语义差异。CPU准备阶段同样优先交付可运行的Trainer CLI配置，不能仅因环境缺少GPU依赖而另造训练循环。
- 读取并验证完整捕获，再构建该模型现有原生data/sequence context。保留label shift、mask、packing、位置和模态语义，不能重新tokenize或丢弃字段。
- 复用模型loss context和全局校准；专用目标放正式loss层，布局信息由model/module提供，更新规则放optim层。
- 由标准engine执行真实forward/backward、累积、裁剪和更新。观测脚本不全局替换模型forward或优化器公式。
- 应用每步实际LR及其他有效组调度；固定峰值LR不等于调度一致。
- 两边独立演化optimizer和参数，不将参考loss、梯度或更新后权重注入目标训练。
- 参数和group配置要来自本次模型契约。已有模型专用回放脚本可参考其编排方式，不能直接作为所有模型入口。

## 5. 分布式拓扑与结果组织

明确哪些ranks共享样本、切序列、切参数、执行pipeline或分发专家。保留同一全局batch后再调整本地micro/累积；实际DP size由数据拓扑决定，不总是world/SP。

检查模型是否支持所选SP/TP/CP/EP以及heads、chunk、sequence等整除条件。用现有mesh和split/gather接口，不另造训练通信实现。

为不同模型/拓扑/精度配置准备独立结果目录；不同JSONL名称不代表completion/config不会同名覆盖。固定短步数完成smoke后，再做用户要求的多步轨迹；profiling默认关闭，需要时短窗口开启。

具体命令从该模型当前CLI生成，列明checkpoint、data、topology和输出。不要从案例复制八卡、固定GBS、Muon或VQ/HLM配置。

启动前先发改动与回放来源报告，覆盖Megatron和xTuner各自的文件/层次、改动目的、数学与默认行为影响，并明确是否仅涉及配置/训练脚本。按字段说明回放来源（如batch、labels/masks、LR/调度、初始checkpoint或optimizer state），区分初始化与逐步回放；列明当前框架独立执行的forward/backward、loss、梯度、裁剪及optimizer更新。参考loss/gradnorm等仅用于比较，不能作为训练输入。没有改动或回放时明确写无；这份启动报告不构成额外启动审批；先完成原实现对比，取得双方结果后再询问是否修改实现。

同一报告列明双方log/metrics、独立TensorBoard run目录和展示命令，以及profiling有效开关（默认关闭）、开启方式与trace目录。先在smoke中确认逐步性能日志和双方TensorBoard事件确实产生，再进行完整实验；未实跑时说明这些仍待GPU验收。

## 6. 验证与归因

先核对所有必要rank/stage完成，step/micro无缺失/重复、全局输入身份、有效targets、实际LR与状态一致，排除skipped/nonfinite。PP只有特定stage产出loss或checkpoint时按真实拓扑检查，不假定所有rank文件完全同构。

分别比较主loss、辅助loss、total、裁剪前norm、必要参数和梯度的canonical布局误差。记录损失日志的分母和权重，不按同名指标直接比较。增加参考自身rerun估计数值漂移；阈值取决于模型、dtype和任务，不固化历史容差。

先定位首步差异，再分析多步轨迹；逐层/逐目标定位首个分歧。CPU primitive测试不证明目标GPU融合或并行backward正确。取得双方有效对比结果后，汇总原实现差异及实测影响、拟修改范围和保留原实现的选项，询问用户是否修改；确认后才实施，并保留修改前结果供后续比较。

## 独立训练与严格回放分开

独立入口使用native数据与本地scheduler，可用于真实训练/性能，但不能仅凭同数据集、tokenizer、seed就做逐步parity声明。相同token文件还可能有不同mixture权重、采样、拼接和边界。

经上述缺口评估后采用的独立外层循环，即使与标准Trainer共享核心更新接口，也必须显式对齐初始化、数据/target、loss、scheduler时点、异常策略、checkpoint和计时边界，才能视为等价实验；共享TrainEngine本身不能替代这一核对。
