# CLM 复现封存包

本目录整理截至北京时间2026年11月17日的 CLM 参数强化学习复现工作。用户在清除 Goal 后要求封存，项目按提前结束记录。

## 先读这几项

| 阅读目的 | 文件 |
|---|---|
| 理解整个执行与训练框架 | [框架说明](FRAMEWORK.md) |
| 了解已获得的结果、失败和局限 | [核心发现](FINDINGS.md) |
| 找源码、模型、数据及原始轨迹 | [资源清单](RESOURCES.md) |
| 检查配置与使用已有代码 | [使用与复现指南](REPRODUCTION.md) |
| 核对封存处置与原计划缺口 | [封存状态](ARCHIVE_STATUS.json) |
| 检查文件完整性 | [公开文件清单](MANIFEST.json)及 `validate_archive.py` |

## 已实证的范围

- C2 条件完成并验收1次真实 Qwen3.5-9B 参数更新，原计划70次。
- 首次更新使用8组、每组32条完整逻辑轨迹，共256条，转换为11,095个逐请求训练片段。首批成功43/256，这是更新前策略产生的数据。
- 第一次更新的有限 loss、梯度、语言权重变化、服务同步，以及真实 Token、logprob、mask 和双通道优势已核验。
- 第二批已完成采样与参考模型计算，参数训练在用户封存时终止。该批未提交第二次更新。
- C1、Summary 对照、额外种子、500题验证选模与830题最终评测仍有待完成。首个每5步保存点尚未到达，实际 optimizer/RNG/checkpoint 恢复未验收。

原训练进程及本项目专用服务已停止。物理GPU4–7均已释放。原始数据保留在阿里云 CPFS，公开仓库提供代码、配置、证据摘要与定位信息。

## 目录

```text
archive/2026-11-17/
├── README.md
├── FRAMEWORK.md
├── FINDINGS.md
├── RESOURCES.md
├── REPRODUCTION.md
├── ARCHIVE_STATUS.json
├── MANIFEST.json
├── FRAMEWORK_MANIFEST.json
├── validate_archive.py
├── framework/
│   ├── source-revisions/training-v10/clm_replication/
│   ├── agent-bundle-v1/clm_replication/
│   ├── src/                     # 实际消费的底座修改文件
│   ├── management/              # 实际入口、配置、锁文件、准备脚本
│   ├── data/                    # 题单来源与选择清单
│   ├── licenses/
│   └── UPSTREAM.json
└── evidence/                    # 真实验收及封存证据
```

历史调查、原计划、工程准备与测试继续保存在仓库的 `docs/`、`plans/` 和 `implementation/`。本目录的封存状态用于解释这些历史材料中的“当前”“准备完成”等时间相关表述。

实际私有运行根为 `/mount-data/CLM-Replication`。公开包不包含 API 凭据、模型权重、已安装的二进制环境和全部大体积原始轨迹。这些资产的来源、哈希及存放位置见资源清单，代码和配置的公开文件范围以清单为准。
