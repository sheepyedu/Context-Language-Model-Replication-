# 使用与后续复现实验

## 本地检查封存包

在仓库根目录运行以下命令。检查器只读取文件及AST，不连接服务器、不调用模型。

```bash
python3 archive/2026-11-17/validate_archive.py
```

结果检查公开manifest、Python语法、JSON结构、核心源文件及封存计数。既有工程CPU检查位于 `implementation/tests/` 和各准备目录，需要对应依赖环境。归档文件检查通过的范围是可读性和一致性。

## 理解调用链

1. 先阅读 `FRAMEWORK.md` 及 `FINDINGS.md`，明确任务成绩、轨迹、Sample和更新的分母。
2. 阅读实际 `framework/management/train-v14.sh` 和 `train-config-v7.yaml`，再读host与agent bundle的对应模块。
3. 用 `bridge.py` 追踪组准入、策略版本与原始Sample，再用 `rewards.py`、`advantages.py` 和原生 `loss.py` 查看优势如何作用于Token。
4. 将实际首批数据验收与原始路径对照，查看 `rollout-0.pt`、`train-0.pt` 和对应32条组结果。读取本方保存的可信 `.pt` 时，在受控CPU环境使用固定Torch版本。
5. 查看Judge原始请求、`judge_rubric.txt` 和canonical账本，核对任务失败、工具/格式失败与基础设施null。

## 在新环境重新搭建

历史入口含原服务器绝对路径、设备布局、服务端口及固定运行ID。移植时在新目录生成配置，保留一份不可变历史快照；不要直接把原运行目录用作新实验输出。

- 先建立隔离Docker、持久存储、明确GPU授权和可用空间，核实际设备UUID、CUDA/驱动及CPU内存。
- 按 `UPSTREAM.json` 和既有方案取得底座源码，叠加本包实际消费文件，核锁文件、模型/tokenizer/template与原始SHA。
- 参考 `bootstrap_agent.sh`、`bootstrap_megatron.sh` 等历史准备脚本建立环境。安装的GPU二进制需要按当前ABI重新核验，安装成功并不等于数值通过。
- 从官方数据源准备题目，核 `data/MANIFEST.json`，明确附件筛选、500题验证及与原论文ID的差异。
- 凭据由目标服务器私有注入。原项目提供 `/mount-data/Tools/skill-api`，公开仓库只保留接口与模型名。
- 为C2/C1/Summary、不同种子和评估生成互斥完整运行ID。现有评估身份修复、rank诊断文件名修复及checkpoint游标修复都有独立CPU证明，正式接入需要重新绑定源码、PINS和入口。
- 先核真实模型输入、工具闭合、Token/logprob/mask、任务和效率优势，再核至少一次真实optimizer及服务同步。候选加速需要真实梯度与实际端到端收益证明。

## 当前停止点与恢复边界

原计划每5次更新保存。封存时仅完成1次，第二批在optimizer提交前被取消，因此没有已验收的完整训练checkpoint。现存初始转换资产缺少本次训练的optimizer/RNG状态。

后续研究应先明确新实验授权、预算和输出目录，再决定如何使用已有闭合数据。原科学请求、失败、UNKNOWN及预算继续保留，不能通过新alias重发旧未知请求。现有 `checkpoint-cursor-prepared-v3` 需要在一个真实完整checkpoint上验证，不能套用到本次缺失的保存点。

恢复标准应覆盖所有rank的模型、optimizer、scheduler、训练RNG、实际数据游标、服务策略版本和已消费组。服务重启的生成RNG没有逐位恢复证明，需要披露。

## 推荐保留的下一轮观测

| 观测 | 目的 |
|---|---|
| 每组成功数、奖励方差来源 | 区分任务学习与格式罚分差异 |
| 编辑发生率与非零效率Token覆盖 | 判断双通道信号是否实际进入训练 |
| 更新前后固定验证集表现 | 判断更新效果，避免以采样成功率代替评估 |
| 各阶段耗时和Token/FLOPs | 定位采样、检索、参考前向、PPO和I/O成本 |
| 完整失败正文与工具原件 | 使基础设施错误能够追溯 |
| 独立运行ID和逐rank数据文件 | 防止跨实验结果复用与文件覆盖 |

原完整科学计划的70步、对照、额外种子、500题选模和830题测试保留在 `plans/subtask-02/`。本次封存不提供这些尚未获得的实验结论。
