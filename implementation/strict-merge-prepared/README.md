# 严格前缀训练合并候选

该目录处于隔离准备状态，生产attempt11继续原逐请求视图。真实Qwen/Megatron完整梯度比较尚待，当前禁止据此宣称实际加速或已上线。

原首批11095逐请求Sample可合并为3401训练区段，训练Token occurrence由134169745降为53681982。原256逻辑轨迹、1705266生成loss Token和177610效率role Token保留。该Token计数降幅约60.0%，其作用是减少重复因果前缀计算。

合并仅允许同组、同会话、同策略、连续trace_index且旧完整输入与生成是下一次输入的精确Token前缀。相邻段奖励、规范化优势、效率优势与截断状态也须相同。工具和中间历史的loss mask与role mask补0，原监督位置的logprob与mask逐值保留。当前候选支持已验证的文本数据，不接受带额外多模态/路由/教师元数据或remove_sample的样本。

合并发生在原任务采样、完整组筛选及轨迹级奖励归一化之后，原生动态batch计数之前。预先归一化后的奖励与role mask绑定哈希，后续标准converter直接复用，避免按区段重新归一化或按首trace覆盖整段role。合并后的显示response为空，完整原始文本由sources定位原轨迹。

## CPU检查范围

- 原生Slime converter前后，所有原生成位置的mask/logprob/奖励/效率项逐值相同。
- 原生动态batch按3401计算，每批仍1次更新，原生DP1 split精确通过。
- 原双通道优势hook对应每个原Token的优势差0。
- 原生Sample序列化roundtrip与篡改拒绝通过。
- 小型因果attention模型含PPO、TIS、KL的全部参数梯度最大差1.39e-17，预设容差1e-12。

现有九项基础CPU及96条原轨迹映射/零生成SGLang前向证据仍保留在服务器原prepared目录。完整Qwen/Megatron梯度、权重更新和实测速率验收之前，不接入正式训练。准备过程中正式训练持续，工程入站gate为空。

## 文件

- strict_merge.py验证并合并训练字段，返回逐原trace的Token区段映射。
- strict_training.py提供候选rollout包装、奖励复用和生命周期接口。
- check_native_integration_r2.py检查原首批和原生CPU入口，包括DP1及真实优势hook。
- NATIVE_INTEGRATION_CPU_R2_ACCEPTED.json记录当前CPU与小模型梯度证明。
