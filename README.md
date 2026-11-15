# CLM 探索与 RL 复现

本仓库保存 Context Language Models 的研究资料、资源清单和参数 RL 实施方案。

| 子任务 | 当前安排 | 入口 |
| --- | --- | --- |
| 一，相关策略和数据清单 | 用户确认完成，提交固定审核 | [资源清单](docs/subtask-01/RESOURCE_CATALOG.md)、[社区复现](docs/subtask-01/COMMUNITY_REPRODUCTIONS.md)、[使用指南](docs/subtask-01/USAGE_GUIDE_ZH.md) |
| 二，原方法 RL 复现 | 方案确定，进入实施 | [完整方案](plans/subtask-02/PLAN.md)、[配置规格](plans/subtask-02/CONFIG_SPEC.json) |
| 三，work 场景数据拓展 | 待定，等待子任务二结果 | 确定范围后补充 |
| 四，work 场景 RL 复现 | 待定，等待子任务二结果 | 确定范围后补充 |

评审采用用户指定的 DeepSeek 4.1f API，配置模型 ID 为 `deepseek-v4-1-flash-261017`。实施时核验实际返回模型。论文原评审为 GPT-5.4-nano，方案记录这一变更对奖励和绝对准确率的影响。

训练采用 Qwen3.5-9B、逐调用 GRPO 与成功门控效率优势。主方案每步 8 题、每题 32 条完整轨迹，共 70 次有效参数更新。十项实施待办当前完成 5/10。当前 attempt05 在原初始策略下持续采样，真实优化器更新为 0/70。实现与环境锁见 `implementation/`，运行中的状态见 `implementation/STATUS.md`。

[Notion 根项目](https://app.notion.com/p/4195e704b6a881838972e810f7ff5744)

[Notion 子任务二完整方案](https://app.notion.com/p/4195e704b6a8819182f3d31f2208a93f)

子任务交付使用固定审核聊天核查，根项目由杨洋审核。资源指南包含历史本机、服务器路径及上游链接，权重和数据通过各自原始入口获取。

## 文档目录

- `docs/subtask-01/` 保存官方资源、社区调查、使用指南和文件结构索引。
- `plans/subtask-02/` 保存 v3 方案、配置规格、来源记录和文档检查。
- 配置的模型 revision、环境锁与镜像摘要在实施阶段解析并冻结。
