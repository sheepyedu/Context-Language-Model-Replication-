# CLM 探索与 RL 复现

本仓库保存 Context Language Models 的研究资料、资源清单和参数 RL 实施方案。

## 2026年11月17日封存

用户已要求封存整个复现任务。代码、核心发现、资源清单、使用指南、Token账目及停止证据集中在 [CLM复现封存包](archive/2026-11-17/README.md)。

本阶段完成并验收1/70次真实参数更新。第二批在提交前按封存要求停止。专属服务器作业已收口，GPU4–7已释放。原始轨迹和大型资产保留于CPFS并提供索引。完整checkpoint恢复、连续更新稳定性、对照、选模与最终评测仍有缺口，封存记录按部分复现处理。

| 子任务 | 当前安排 | 入口 |
| --- | --- | --- |
| 一，相关策略和数据清单 | 用户确认完成，提交固定审核 | [资源清单](docs/subtask-01/RESOURCE_CATALOG.md)、[社区复现](docs/subtask-01/COMMUNITY_REPRODUCTIONS.md)、[使用指南](docs/subtask-01/USAGE_GUIDE_ZH.md) |
| 二，原方法 RL 复现 | 提前结束并封存，1/70次更新已验收 | [封存包](archive/2026-11-17/README.md)、[原完整方案](plans/subtask-02/PLAN.md)、[配置规格](plans/subtask-02/CONFIG_SPEC.json) |
| 三，work 场景数据拓展 | 待定，等待子任务二结果 | 确定范围后补充 |
| 四，work 场景 RL 复现 | 待定，等待子任务二结果 | 确定范围后补充 |

评审采用用户指定的 DeepSeek 4.1f API，配置模型 ID 为 `deepseek-v4-1-flash-261017`。实施时核验实际返回模型。论文原评审为 GPT-5.4-nano，方案记录这一变更对奖励和绝对准确率的影响。

原训练计划采用 Qwen3.5-9B、逐调用 GRPO 与成功门控效率优势，每步8题、每题32条完整轨迹，共70次有效参数更新。封存时科学待办完成5/10，attempt13已停止，真实更新验收1次。实现与环境锁见 `implementation/`，封存状态以 `archive/2026-11-17/ARCHIVE_STATUS.json` 为准。旧文档中的阶段状态保留其历史时间范围。

[Notion 根项目](https://app.notion.com/p/4195e704b6a881838972e810f7ff5744)

[Notion 子任务二完整方案](https://app.notion.com/p/4195e704b6a8819182f3d31f2208a93f)

子任务交付使用固定审核聊天核查，根项目由杨洋审核。资源指南包含历史本机、服务器路径及上游链接，权重和数据通过各自原始入口获取。

## 文档目录

- `docs/subtask-01/` 保存官方资源、社区调查、使用指南和文件结构索引。
- `plans/subtask-02/` 保存 v3 方案、配置规格、来源记录和文档检查。
- `archive/2026-11-17/` 保存本次封存的框架、结论、资源、使用与核验入口。
- 配置的模型 revision、环境锁与镜像摘要在实施阶段解析并冻结。
