# 跨模型数学与并行语义

## 先用原实现运行对比，再确认修改

本参考中的核对项用于发现和解释差异，不是发现后自动修改的授权。先保留两边原实现，启动前说明已知差异和所选共同模式；在已授权环境中直接运行对比，不在结果产生前询问是否统一实现。CPU-only或由用户自行运行GPU的任务交付相应指令，遵守原执行边界。

取得双方可比结果后，再向用户报告并明确询问是否修改，至少包括：

- 双方文件/函数位置、当前规则或计算公式，以及能够证明差异的源码依据。
- 本次有效输入、初始化、调度及完成情况核验；实际loss、梯度/gradnorm、必要参数更新差异，以及参考自身rerun的漂移。
- 差异的性质：例如optimizer参数分组改变更新规则，标量布局影响decay分类，dtype/距离算法改变舍入或离散选码；不能统称为“误差稍大”。区分实测差异、已验证归因与尚未验证的影响。
- 拟修改哪一边、哪些文件/配置和默认行为，并提供继续保留原实现的选项。

没有双方可比结果时报告缺失或阻塞，不以静态推测代替实测并提前请求修改。获得修改确认前继续保留现有实现，不写入修正或通过新增开关启用另一套数学。已有明确确认仅适用于其具体范围；新发现的差异仍先运行对比，再根据结果询问。

## 模型算子契约

先列出两边实际模块与对应关系，检查层序、residual placement、activation、normalization参数化与eps、attention scale、位置编码、mask、dropout和各任务头。对MoE、多模态、SSM或辅助预测分支增加对应项，不把dense文本模型检查表当作完整契约。

区分数学等价、配置一致、容差内数值一致与bitwise相同。shape或参数名相同不证明运算相同，fused/拆分模块也可能通过布局映射实现相同函数。

## 目标函数逐项推导

为每个实际启用目标j记录：numerator N_j、denominator D_j、weight w_j、mask/target、detach位置、局部/全局归约和梯度注入方式。一般写为各项 w_j*N_j/D_j 的组合，不能假定所有辅助目标都除同一个token数。

- 自回归CE：label shift次数、ignore index、有效vocabulary/padded rows、token/sample/square reduction、sample weight及packed边界。
- MoE：router balancing/z-loss、top-k、capacity/drop policy、group统计、shared专家、router bias更新；日志值与autograd缩放可能不同。
- 多模态：哪些位置预测语言、视觉/音频目标如何对齐、模态token mask、patch/frame数量和样本权重。
- 额外分支：如MTP、蒸馏、VQ/HLM或重建目标，仅在该模型启用时核对其target、权重和梯度路径。
- 非可微更新：router bias、EMA teacher等若存在，也属于训练状态契约，不能只比较optimizer参数。

明确哪些all_reduce只记录统计，哪些是可微归约，哪些由引擎在backward后补全缩放。跨microbatch先做局部平均再平均，不一定等于按全局有效target归一化。使用不等长度/有效target数的样本发现此类错误。

## 优化器契约

从实际配置和实现推导更新式，不由名称猜测：Adam/AdamW、Muon或其他optimizer可能有不同momentum、bias correction、epsilon位置和decoupled decay。

核对全部trainable参数与state：canonical名字、算法、LR multiplier、decay、clip资格、冻结参数、共享参数去重、master/state dtype及加载状态。根据语义区分norm、bias、embedding/head、路由表、专家矩阵和fused投影，不只按物理ndim猜组。

如果使用矩阵正交化更新，核对逻辑矩阵切分、scale、momentum、NS系数/次数/eps、cast与通信顺序；不使用该算法的模型不引入专用适配。经用户确认需要实现的新变体复用原通信与标准能力，并以opt-in形式保持其他模型默认行为；opt-in不能替代修改前确认。

gradnorm应对唯一参数梯度在正确group上求norm。先确认当前gradient scale，再比较裁剪系数、更新和state；相同norm不证明方向一致。混合算法切片避免重复decay或重复clip。fresh optimizer测试不证明resume state正确。

## 任意并行拓扑

区分数据DP、TP、PP、CP/SP、EP以及FSDP/HSDP参数分片mesh。EP有时复用DP维度，不能把所有标签简单相乘或用world除某个维度推断数据DP。

当每个数据DP rank每micro处理MBS个唯一样本且无特殊调度时，GA=GBS/(MBS*data_DP_size)；其他batch/packing语义按实际实现定义。PP microbatch调度和virtual stage不应被当成额外数据副本。

SP/CP对齐步骤：

1. 确认同组共享哪些完整样本，native sampler只按数据DP rank分发；验证完整输入身份。
2. 用现有sequence split/gather和模型接口，保留位置、文档边界和该模型要求的padding/chunk整除条件。
3. labels、loss mask和辅助target应与消费它们的局部/完整特征对应，边界目标不能在切分点丢失或重复。
4. 为每个目标确定是分片计算还是副本计算，再推导denominator、可微collective backward和FSDP/DP平均的组合。不能一律除SP size，也不能一律WORLD SUM。
5. 日志只统计唯一数据/targets，保留原样本和micro归属；区分用于训练的归一化与仅用于展示的去重。
6. PP/TP/EP下用正确stage/group记录与比较。不能对仅部分rank参与的目标直接发WORLD collective造成死锁。

用不同group大小、不等有效长度和多个microbatch验证loss值、梯度方向/尺度、参数更新及共享参数去重。CPU/Gloo primitive测试不替代NCCL、模型真实gather backward和GPU kernel。

## 精度与数值路径

按模型记录参数存储、计算、master、梯度累积/归约、optimizer state和中间算子dtype；FP32/BF16/FP16/FP8以及量化状态均从实际配置发现。

- 混合精度与loss scaling：动态scale、overflow判定、unscale/clip/skip顺序一致。
- GEMM与归约：梯度直接FP32累积或低精度生成再cast、foreach/fused更新、分块GEMM、all-reduce/reduce-scatter顺序。
- norm/softmax：eps定义、zero-centered gamma、计算前cast、fused backward和compile引入的舍入边界。
- attention/SSM/MoE：真实backend及版本、accumulator、routing/tie-breaking、capacity、scan/chunk、padding和通信调度。
- compile/recompute：真实target、graph break、RNG preservation；固定seed不能保证不同实现消耗相同随机数序列。
- FP32保存pattern可能还改变运行时sharding/dtype，需读当前实现，不能只看全局param_dtype。

数学算式一致但GPU融合路径不同，应报告数值差异并按合理容差验证，而非宣称bitwise；不能为消除差异静默改变参考模型行为。

## 特定模型适配

Qwen3.5-9B ConceptLM的VQ/HLM、MegatronMuon矩阵切分、SP副本校准和可选LM z-loss方案见 [qwen35-conceptlm-example.md](qwen35-conceptlm-example.md)。仅作为具体案例，不将其loss权重、八卡拓扑或更新参数推广到其他模型。
