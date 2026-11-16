# Summary 对照运行器准备

2026-11-16。该目录用于方案中的 S0 未训练基线和 S1 仅任务奖励训练。当前状态为隔离实现及 CPU 接线通过，尚无真实 S0/S1 任务或参数更新，T09 仍未完成。

## 实现与默认选择

官方固定源码只提供 CLM harness。本实现复用其任务和 Bash 工具说明、持久 shell、工具参数解析、工具返回格式和独立提交规则。Summary 的 system prompt 由原公共前部、自动摘要说明及原提交说明组成。这是明确选定的复现默认值，作者的 Summary 完整实现未公开。

保留原问题、Web 工具说明和 answer.txt 交付路径。上下文按原模型模板计数，到 28672 tokens 时调用同一策略生成摘要。原 system 和任务消息保留，先前交互被摘要替换。服务窗口 32768，单次输出最多 4096，最多 100 个任务回合和 224 个总模型调用。摘要调用消耗模型预算并保存请求、响应及用量，作为 task-only 生成参与训练。所有效率 role mask 为 0，S1 在真实训练 actor 中还须明确验证效率权重为 0。

`summary_runner.py` 通过与原 agent 相同的 LiteLLM 路由调用模型，temperature .7、top_p .95、thinking true、无自动重试。模型和工具异常保留原输入，UNKNOWN 不重复发送。每个 episode 先用独占 INTENT 防止同一或重建对象重新运行。摘要的 length、空文本、工具输出和超长结果均保存为负结果，不自动重新生成。

本地计数器在副本中将 assistant tool arguments 的有效 JSON 字符串解析为对象，符合原 SGLang 服务的预处理，再应用相同 tokenizer/template。线上的请求和原始参数保持原文。使用标准库 JSON 解析并显式拒绝 NaN/Infinity。该兼容证明覆盖本接口支持的 JSON 参数，未宣称两个不同 JSON 库在任意非标准输入下完全相同。

## 目录与部署边界

- `summary_loop.py` 负责任务、摘要与工具回合和全量留档。
- `summary_runner.py` 是任务容器入口，加载固定模板与原生 Bash 环境。
- `summary_polar_agent.py` 只替换原宿主任务命令的 Python 入口、隔离依赖路径及 tokenizer 参数。
- `summary_context.py` 来自先前已验收的隔离组件，固定 Codex 摘要提示和来源在 `summary-prepared/prompts`。
- `agent-bundle-summary-v4` 是远端容器只读包。宿主 `summary-host-source-v4` 精确复制当前 training-v10 的 16 个文件并增加 SummaryHarness，供之后的专用 coordinator/gateway 使用。
- `s0-task-config.yaml` 和 `s1-task-config.yaml` 保留原 per_request builder、DeepSeek evaluator、镜像和沙箱资源，新增只读 tokenizer/依赖挂载，移除可选的普通文件暂停 gate。

运行中的 C2 包、gateway、任务和模型配置没有使用本目录。后续启用须先在原任务闭合边界准备独立 owner/gateway，冻结完整来源与模型，并验证真实 Polar 请求、token/logprob/mask、摘要事件和 evaluator 对齐。当前 YAML 只是任务配置，不是独立完整训练启动命令。

依赖采用当前 RL 环境中已存在的 Transformers5.3.0、tokenizers0.23.0rc0 ABI3、safetensors0.9.0rc1 ABI3，放在隔离 target。真实 agent Python3.12 容器已加载固定 Qwen tokenizer。完整依赖和模型模板文件清单见 `ASSETS_PREPARED.json`。

## 实际验收与保留失败

| 验收 | 实际结果 |
| --- | --- |
| 原生任务镜像、UID1005、2CPU/4GiB、无网络无GPU、只读源 | R4 真实容器 exit0，20/20 |
| 持久 Bash、提交、多工具首个执行、原始返回、异常一次消费 | 通过，全部 mock completion |
| 完整入口、原 SDK 参数及真实模板越阈值摘要 | 通过，真实工具执行、模型响应为桩 |
| 宿主 instruction delivery 与原命令差异逆变换 | 4种输入精确一致，包括多行和 shell 字符 |
| 原生 PerRequestBuilder 与 DeepResearchEvaluator | 4/4 宿主检查，通过3条合成 completion 的摘要计数/事件绑定/0效率 mask，Judge 为桩 |
| 真实模型生成、真实 S0/S1 rollout | 0，仍待合法运行边界 |

R1 14/15 暴露 UNKNOWN episode 重入，R2 16/17 暴露工具 arguments 字符串与 HF 模板对象预期不一致，R3 在 import 时发现容器缺 orjson。各失败容器均已原整数 exit1 收口并保留源、日志和回执，独立新版本修复。宿主 R1 因只读容器包缺宿主插件导入失败，随后分离出完整宿主包，R2 四项通过。原生产 C2 持续运行，新增真实模型、工具环境任务或 Judge 调用均为 0。

原始证明在远端 `management/summary-harness-prepared-v2`，原失败目录保留。CPU 结果没有验证模型摘要质量、真实路由接线或完整参数 RL 稳定性。
