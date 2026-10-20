# 模型无关的数值诊断与性能分析

## 证据优先

读取原始metrics、completion/exit/status、有效launch/config、执行源码和软件环境。仅容忍正在写入的末尾不完整JSON，中间损坏应报错。分清训练完成、checkpoint完成和后处理/上传失败，最终非零退出码不必然表示训练中断。

先看原始指标再看可视化。校准iteration/step基数、micro和update、日志采样间隔、loss分母和副本去重。模型不同不保证指标名和文件schema相同。

## 数值差异定位顺序

1. 核对初始状态、完整输入、所有目标的target/mask、LR和并行契约；先排除不同实验。
2. 首次更新前已有forward/gradnorm差异时，优化器更新不能解释最初偏差；优先核对权重、前向/反向、归一化及norm统计。
3. 比较参考自身rerun的漂移；区别固定倍数、特定样本异常和累积轨迹分离。
4. 在canonical布局比较分量loss、逐参数gradnorm、相对L2/绝对误差、cosine和更新。对接近零量使用绝对误差。
5. 按任务检查异常输入：文本重复与packing、图像尺寸/patch、音频时长、有效target数量、专家routing/capacity等。异常内容相关不是充分因果证据，不自动修改数据。
6. 固定同初始状态与输入、不更新optimizer，逐目标和逐层定位首个分歧；再对后端、dtype、融合、compile或通信做单变量诊断。
7. 对有随机路由/dropout/增强的模型，先对齐相应输入/状态或明确不确定性；不通过强制复制参考中间结果制造最终parity。

诊断hooks只读取不改变tensor，保持在独立测试/工具层。模型特有的历史线索见 [qwen35-conceptlm-example.md](qwen35-conceptlm-example.md)，不能当成新模型根因。

## 梯度裁剪与尖峰

裁剪前norm大不意味着参数更新同比大。统一尺度差可能被裁剪抵消；少数参数异常会因全局裁剪压低正常部分，方向和状态差异仍存在。还需考虑optimizer的尺度不变性、momentum和自适应state。

有限尖峰不等于已发散或最终质量变差；总gradnorm不足以区分尺度和方向。记录裁剪前norm、系数、分组贡献和必要的state/更新误差。

## 双方日志与TensorBoard

Megatron捕获/参考rerun和xTuner回放/独立训练均启用训练与性能记录。优先复用各自原生logger、timer和TensorBoard writer；缺少指标时在训练观测层补齐，不改loss、梯度或optimizer数学。每个optimizer update至少在文本log打印step、每步耗时（ms或s）、全局TPS、TPS/GPU，并在结构化metrics和TensorBoard中记录同一步值。区分micro-step与update；有skipped update时显式标记。MFU有可靠模型FLOPs及峰值口径时同时记录，否则说明不可得，不伪造估计。

双方TensorBoard至少包括各项loss/total、裁剪前gradnorm、实际LR、每步耗时、TPS和TPS/GPU，模型不适用的指标注明原因。保存原始tag与统一比较tag的对应关系、单位、step基数和计时边界；不能为了同名曲线改变原始数学口径。每个模型/框架/配置用独立event目录，参考捕获、Megatron rerun和各xTuner配置不覆盖。

交付实际event路径及展示命令，例如将共同父目录组织为独立runs后运行：

```bash
tensorboard --logdir "$COMPARISON_TENSORBOARD_ROOT" --port "$TB_PORT"
```

不在CPU准备阶段自动启动服务。GPU验收需检查双方event文件中的实际scalar、step覆盖与原始log/metrics一致，确认可以同屏显示；只创建目录或给出命令不代表已记录。性能字段计算口径如下，动态/多模态任务须同时说明TPS所计token及实际工作量。

## 性能口径

按相同模型工作量、相同步骤窗口比较，排除启动编译与显式profiler active步骤，保留mean、median和异常值。不同计时边界要披露：取数据/预处理、H2D/context、forward/backward、clip/optimizer、通信、日志/checkpoint是否计入，以及是否CUDA同步、取最慢rank。

固定长度文本且每条样本长度相同的情况：

```text
processed tokens/update = GBS * full sequence length
aggregate TPS/GPU = 步数 * GBS * full sequence length / 总耗时 / world
```

这包含masked位置，不是effective-target TPS，也不是每步TPS的算术平均。变长/多模态应使用逐步真实token、patch/frame/audio或样本工作量；不能将固定GBS*seq公式无条件套用。MoE的active FLOPs与dense FLOPs也不同。

已有工具仅用于固定长度文本和当前兼容metrics/log schema，从仓库根运行：

```bash
python .claude/skills/xtuner-megatron-alignment/scripts/summarize_performance.py \
  --xtuner-metrics /path/to/metrics.jsonl \
  --megatron-log /path/to/rank0.log \
  --global-batch-size "$GBS" --sequence-length "$SEQ_LENGTH" \
  --world-size "$WORLD_SIZE" --skip-first "$STARTUP_STEPS"
```

--exclude-step可重复排除零基step。工具只读、输出重叠/未匹配步骤与聚合TPS，不验证训练完成或精度。不同模型schema或动态工作量时扩展解析与测试，而不是伪造相同字段。

MFU需要该模型正确FLOPs公式、实际工作量、精度和硬件峰值；没有真实记录就写未记录，不能由TPS随意推算。注明显存的rank、allocated/reserved、是否包含编译、每步峰值或累计峰值。

## 优化与profiling

先读实际配置/trace，再决定优化：

- 参数驻留、sharding、all-gather预取/重叠、梯度累积同步次数、PP bubble与TP/CP/EP通信。
- 实际模型的算子融合、稀疏routing/dispatch、attention/SSM/backend、loss分块和optimizer内核。
- compile覆盖、recompute、offload、内存分配和data pipeline瓶颈。

相同标签的ZeRO/FSDP选项可能只控制一个驻留设置，解释底层真实行为；不将某个入口--zero-stage映射当所有模型定义。no_sync的额外梯度驻留和参数保留需要显存验证，不保证所有拓扑更快。

Megatron与xTuner都接入profiling，优先复用各自原生profiler；两边默认关闭，仅显式启用时创建profiler并输出trace。提供enable、wait/warmup/active或start/end step、rank选择及独立trace目录等实际支持的配置，明确step是micro还是update，不虚构统一CLI。显式短窗口采集并标明rank/group/step，确保异常和正常退出时正确停止/关闭。检查默认关闭时不生成trace、不执行逐步profiler同步或导出；开启时确有有效窗口的CPU/CUDA算子与相关通信trace，不因有profiler代码或TPS日志就认为已采集。没有trace不量化kernel/通信贡献；未验证autograd的前向融合不能直接用于训练。性能优化不得静默改变模型、目标或精度契约。
