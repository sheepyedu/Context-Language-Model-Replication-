# 实施状态

观测时间为北京时间2026-11-16 01:38。待办1/10，有效优化器更新0/70。原参数RL目标继续。

Qwen3.5-9B因果语言参数8,953,803,264已转换为TP2 Megatron检查点，独立embedding和lm_head保持。仅使用用户授权GPU4/5。输入、奖励、mask与Token账本CPU检查已通过，真实模型工具接线已有一份完整正例和一份工程失败，不能当RL训练成绩。

attempt01因VLM processor要求会话形式输入而退出。无损会话封装已核3540题原文字段一致。attempt02因memory-saver预加载选错CUDA运行库退出，已通过RED/GREEN和GPU4/5真实Ray张量检查。attempt03完成SGLang模型加载后，因显式外部router地址而未创建router，退出1。attempt04增加自有router生命周期，后端初始化进行中。

所有失败原件保留，正式训练模型调用及优化器更新目前均0。T02还需初始权重同步后固定输入logprob校验。训练稳定门为连续3个已验收的真实更新。

# Token统计和轨迹保留

每次rollout和DeepSeek-V4.1-Flash评审都生成不可重复的TOKEN_RECEIPT，并纳入全局账本。候选丢弃、失败、验证、对照及工程调用同样记账。输入含cache子集，输出含thinking子集，两者不重复相加。缺失usage保留null。训练样本被重复读取的Token量单列。

目前工程调用共37次，rollout34次，输入234208、输出6192。评审3次，输入2124、输出737。已知总Token为243261。旧首个成功工程样本的部分临时辅助文件被上游cleanup删除，完整模型轨迹仍在，缺失已登记。新cleanup会在删除临时目录前逐文件归档和hash。

CPFS约6.75TB可用，当前选择保留全部rollout，包括请求、响应、token IDs/logprob、工具调用和返回、上下文镜像前后及评审。最终至少交付3组完整轨迹。内部服务warmup仅按服务日志单列，未冒成具有完整usage回执的业务调用。

完整原件位于服务器 `/mount-data/CLM-Replication/runs`。本仓库保留实现与必要小证据，模型、原数据、API凭据及大轨迹归档保存在各自专用路径。
