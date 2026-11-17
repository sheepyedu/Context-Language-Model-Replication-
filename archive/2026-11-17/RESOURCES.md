# 有效资源清单

## 本仓库

| 资源 | 路径 | 用途与验收范围 |
|---|---|---|
| 封存冻结源码 | `framework/source-revisions/training-v10/clm_replication/` | 实际消费的执行、奖励、轨迹转换与训练适配 |
| 实际agent bundle | `framework/agent-bundle-v1/clm_replication/` | 容器内harness接线，独立于host模块版本 |
| 原生底座修改文件 | `framework/src/` | 实际Slime/ProRL消费者与双通道接线 |
| 正式入口 | `framework/management/train-v14.sh`、`owned_training_v14.py` | attempt13实际入口和原退出保留 |
| 软件与源锁 | `framework/management/*packages.lock`、`SOURCE_PINS.training-v14.json` | 4,111项代码/环境资产指纹，GPU4–7条件 |
| 原计划及默认值 | 仓库 `plans/subtask-02/` | 论文、官方、底座默认与项目选择的区分 |
| 原资源调查 | 仓库 `docs/subtask-01/` | 官方资源、目录用途、社区复现及使用说明 |
| 逐Token验收 | `evidence/FIRST_UPDATE_DATA_ACCEPTED.json` | 256轨迹/11,095 Sample实审 |
| 参数变化与同步 | 仓库 `implementation/management/FIRST_UPDATE_AND_CHAIN_V2.md`等 | 第一次更新、原日志协调及policy1锚 |
| 科学负结果 | 仓库 `implementation/evidence/STRICT_MERGE_NATIVE_V6_ROOT_REJECTED.json`及重计算拒绝记录 | 保留未通过候选与原判据 |
| 后续准备 | 仓库 `implementation/management/`及Summary/C1目录 | checkpoint游标、评估身份、检索驱动、对照及数据审计准备，正式使用状态见各README |
| 封存停止证据 | `evidence/STOP_RESULT.json` | 原owner退出、专属进程收口与GPU释放 |

`FRAMEWORK_MANIFEST.json`给出公开冻结代码的逐文件SHA。大型已安装环境资产保留在原服务器，锁文件与完整source pins提供定位和指纹。准备版本、已拒绝版本、实际消费版本各有原标签。

## 官方资源

- 论文正文及附录提供算法、RL参数、实验和局限。原始研究引用固定v1。 [论文](https://arxiv.org/html/2609.37725v1)
- 官方仓库包括CLM harness、ICL、RL README及两个RL接线补丁，归档依据固定提交 `483198229ab7b90c0eed4651306ab3b3505aaf50`。 [官方仓库](https://github.com/facebookresearch/context-language-models/tree/483198229ab7b90c0eed4651306ab3b3505aaf50)
- 上下文任务和配套Skills的资源清单见原调查文档，服务器完整快照位于 `/mount-data/context language model`。固定移植版与其差异说明见原调查记录。 [ContextBench](https://github.com/RulinShao/ContextBench/tree/fb533c18c78a82d3f1b2047a9c1700735cf829ae)
- 参数RL接线以CLM所指ProRL旧奖励语义为基础，固定recipe提交 `8bc67cc32607c31988406a2de460b350970efc88`。 [ProRL](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/tree/8bc67cc32607c31988406a2de460b350970efc88)
- 固定Slime recipe提交 `bf9b1a3bc5dd493000ae8518f737fb606bbe36e7`，本项目修改文件及指纹随封存包保存。 [Slime](https://github.com/THUDM/slime/tree/bf9b1a3bc5dd493000ae8518f737fb606bbe36e7)
- 官方基座模型下载源、文件大小及SHA见 `framework/management/MODEL_ORIGIN.json`。下载时解析的是ModelScope master，实际内容以已固定文件SHA核对。 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- 训练原始rl分割3,051题，经附件引用筛选保留3,040题；验证从独立sft分割固定种子1234选500题，标准化问题无重合，原论文精确ID未确认。 [OpenResearcher-Dataset](https://huggingface.co/datasets/wAI-org/OpenResearcher-Dataset)
- 最终测试资产为830题和100,195篇固定语料，检索准备已完成，正式830题评估未执行。 [BrowseComp-Plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus)

远端源码通过归档解包部署，原服务器依赖目录没有完整Git元数据。当前冻结文件SHA能够核对实际代码，Megatron的精确原Git revision未由本次导出重新确证，软件版本见锁文件。

## CPFS原始证据

运行根为 `/mount-data/CLM-Replication`，服务器项目为skill-RSI，持久化文件保留。

```text
runs/c2-seed1234/attempt-13/              原入口、controller日志、原生退出
runs/c2-seed1234/training-attempt13/      rollout-0.pt、rollout-1.pt、train-0.pt
runs/c2-seed1234/audit/                  原始候选、组结果、顺序与数据游标
runs/provider-attempts/                 逐模型请求、原响应及Token回执
runs/session-artifacts/                 任务文件、工具、镜像编辑事件和归档manifest
runs/session-artifacts/*/logs/judge/     DeepSeek评审的原始请求与回执
runs/token-ledger/                      去重账本
runs/token-reports/1794903004443684138/   最后完整维度对账及CSV
runs/web-retrieval/                     已回传的检索原文与回执
management/archive-20261117/            本次导出与停止的原证
checkpoints/qwen35-9b-initial/           起始模型转换资产
```

初始转换checkpoint元数据中的iteration1不表示一次RL更新。正式训练checkpoint未完成。原始模型在 `/mount-data/Models/Qwen3.5-9B`，原官方资产在 `/mount-data/context language model`，共享数据集在 `/mount-data/Datasets`。

公开仓库保留可分发的代码与核验摘要。原始会话、完整网页内容、模型权重、安装环境二进制及凭据继续由服务器权限管理，未混入公开Git文件。大型原件的当前清单和历史哈希范围在封存状态中说明，不能由文件存在推导训练可恢复。
