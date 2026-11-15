# CLM 源码逐文件导航

对应上传的固定版本。核心入口详解见 [使用指南](USAGE_GUIDE_ZH.md)。代码文件逐项列出，256 个生成任务按相同结构说明。部分功能说明来自模块 docstring 和顶层定义，实际参数以源码为准。

## context-language-models

| 文件 | 用途 |
|---|---|
| [.gitignore](repositories/context-language-models/.gitignore) | 仓库工具或版本控制配置 |
| [CODE_OF_CONDUCT.md](repositories/context-language-models/CODE_OF_CONDUCT.md) | 该主题的文档，文件名标识具体主题 |
| [CONTRIBUTING.md](repositories/context-language-models/CONTRIBUTING.md) | 该主题的文档，文件名标识具体主题 |
| [LICENSE](repositories/context-language-models/LICENSE) | 许可证 |
| [NOTICE](repositories/context-language-models/NOTICE) | 版权与归属声明 |
| [README.md](repositories/context-language-models/README.md) | 所在模块说明与使用方法 |
| [assets/teaser.png](repositories/context-language-models/assets/teaser.png) | 图形、论文插图或文档附件 |
| [clm/clm_contextbench/README.md](repositories/context-language-models/clm/clm_contextbench/README.md) | 所在模块说明与使用方法 |
| [clm/clm_contextbench/__init__.py](repositories/context-language-models/clm/clm_contextbench/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_contextbench/clm_stream.py](repositories/context-language-models/clm/clm_contextbench/clm_stream.py) | ContextBench 流式协议与 CLM 接线 |
| [clm/clm_harness/README.md](repositories/context-language-models/clm/clm_harness/README.md) | 所在模块说明与使用方法 |
| [clm/clm_harness/__init__.py](repositories/context-language-models/clm/clm_harness/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/agent_trajectory_format/README.md](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/README.md) | 所在模块说明与使用方法 |
| [clm/clm_harness/agent_trajectory_format/__init__.py](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/agent_trajectory_format/builder.py](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/builder.py) | 轨迹构建 |
| [clm/clm_harness/agent_trajectory_format/harness_export.py](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/harness_export.py) | 运行器轨迹导出 |
| [clm/clm_harness/agent_trajectory_format/models.py](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/models.py) | 轨迹数据模型 |
| [clm/clm_harness/agent_trajectory_format/validator.py](repositories/context-language-models/clm/clm_harness/agent_trajectory_format/validator.py) | 轨迹结构验证 |
| [clm/clm_harness/cli.py](repositories/context-language-models/clm/clm_harness/cli.py) | 命令行接口 |
| [clm/clm_harness/clm_agent/__init__.py](repositories/context-language-models/clm/clm_harness/clm_agent/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/clm_agent/harness.py](repositories/context-language-models/clm/clm_harness/clm_agent/harness.py) | 执行主循环，模型调用、工具、上下文同步与输出 |
| [clm/clm_harness/clm_agent/prompts.yaml](repositories/context-language-models/clm/clm_harness/clm_agent/prompts.yaml) | 系统提示模板 |
| [clm/clm_harness/configs/bcp.yaml](repositories/context-language-models/clm/clm_harness/configs/bcp.yaml) | 该模块/任务的配置或元数据 |
| [clm/clm_harness/configs/edgebench.yaml](repositories/context-language-models/clm/clm_harness/configs/edgebench.yaml) | 该模块/任务的配置或元数据 |
| [clm/clm_harness/context_env/__init__.py](repositories/context-language-models/clm/clm_harness/context_env/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/context_env/edit_gate.py](repositories/context-language-models/clm/clm_harness/context_env/edit_gate.py) | 上下文编辑接受条件 |
| [clm/clm_harness/context_env/env.py](repositories/context-language-models/clm/clm_harness/context_env/env.py) | 模块说明 · ContextEnv — writable-context OS around a Harbor sandbox. |
| [clm/clm_harness/context_env/paths.py](repositories/context-language-models/clm/clm_harness/context_env/paths.py) | 上下文文件路径 |
| [clm/clm_harness/context_env/types.py](repositories/context-language-models/clm/clm_harness/context_env/types.py) | 上下文环境数据类型 |
| [clm/clm_harness/context_utils/__init__.py](repositories/context-language-models/clm/clm_harness/context_utils/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/context_utils/context_string.py](repositories/context-language-models/clm/clm_harness/context_utils/context_string.py) | 上下文文本与消息结构转换 |
| [clm/clm_harness/flops_metrics/__init__.py](repositories/context-language-models/clm/clm_harness/flops_metrics/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/flops_metrics/kv_cache_flops.py](repositories/context-language-models/clm/clm_harness/flops_metrics/kv_cache_flops.py) | 考虑缓存复用的计算量核算 |
| [clm/clm_harness/task_templates/__init__.py](repositories/context-language-models/clm/clm_harness/task_templates/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/task_templates/edgebench.yaml](repositories/context-language-models/clm/clm_harness/task_templates/edgebench.yaml) | 该模块/任务的配置或元数据 |
| [clm/clm_harness/task_templates/open_problems.yaml](repositories/context-language-models/clm/clm_harness/task_templates/open_problems.yaml) | 该模块/任务的配置或元数据 |
| [clm/clm_harness/task_templates/terminal_agent_tasks.yaml](repositories/context-language-models/clm/clm_harness/task_templates/terminal_agent_tasks.yaml) | 该模块/任务的配置或元数据 |
| [clm/clm_harness/utils/__init__.py](repositories/context-language-models/clm/clm_harness/utils/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_harness/utils/budget.py](repositories/context-language-models/clm/clm_harness/utils/budget.py) | 上下文预算、提醒和溢出处理 |
| [clm/clm_harness/utils/finish_policy.py](repositories/context-language-models/clm/clm_harness/utils/finish_policy.py) | 任务结束策略 |
| [clm/clm_harness/utils/lm_compat.py](repositories/context-language-models/clm/clm_harness/utils/lm_compat.py) | 模型请求兼容处理 |
| [clm/clm_harness/utils/pricing.py](repositories/context-language-models/clm/clm_harness/utils/pricing.py) | 成本价格统计 |
| [clm/clm_harness/utils/provenance.py](repositories/context-language-models/clm/clm_harness/utils/provenance.py) | 代码来源与版本指纹 |
| [clm/clm_harness/utils/resume.py](repositories/context-language-models/clm/clm_harness/utils/resume.py) | checkpoint 恢复与历史回放 |
| [clm/clm_harness/utils/skills.py](repositories/context-language-models/clm/clm_harness/utils/skills.py) | Skill 文件加载、工具挂载和提示注入 |
| [clm/clm_harness/utils/tokens.py](repositories/context-language-models/clm/clm_harness/utils/tokens.py) | token 统计和快照工具 |
| [clm/clm_harness/utils/tool_schemas.py](repositories/context-language-models/clm/clm_harness/utils/tool_schemas.py) | 工具接口 schema |
| [clm/clm_icl/README.md](repositories/context-language-models/clm/clm_icl/README.md) | 所在模块说明与使用方法 |
| [clm/clm_icl/__init__.py](repositories/context-language-models/clm/clm_icl/__init__.py) | Python 包初始化与公开接口 |
| [clm/clm_icl/evolve.py](repositories/context-language-models/clm/clm_icl/evolve.py) | Skill 优化循环、评估与候选选择 |
| [clm/clm_icl/note.py](repositories/context-language-models/clm/clm_icl/note.py) | 对比样本选择和 proposer 证据 |
| [clm/clm_icl/propose.py](repositories/context-language-models/clm/clm_icl/propose.py) | 候选模型调用、解析与校验 |
| [clm/clm_icl/proposer_prompt.md](repositories/context-language-models/clm/clm_icl/proposer_prompt.md) | Skill 编辑模型提示词 |
| [clm/clm_icl/tasks.py](repositories/context-language-models/clm/clm_icl/tasks.py) | 任务接口、运行结果与轨迹摘要 |
| [clm/clm_rl/README.md](repositories/context-language-models/clm/clm_rl/README.md) | 所在模块说明与使用方法 |
| [clm/clm_rl/patches/0001-slime_bridge-dual-channel-advantage.patch](repositories/context-language-models/clm/clm_rl/patches/0001-slime_bridge-dual-channel-advantage.patch) | 外部训练框架的版本化代码补丁 |
| [clm/clm_rl/patches/0002-slime-forward-a_eff-and-role_masks.patch](repositories/context-language-models/clm/clm_rl/patches/0002-slime-forward-a_eff-and-role_masks.patch) | 外部训练框架的版本化代码补丁 |
| [clm/examples/run_harbor.sh](repositories/context-language-models/clm/examples/run_harbor.sh) | 通用 Harbor 单题启动 |
| [clm/tests/test_context_string.py](repositories/context-language-models/clm/tests/test_context_string.py) | 所在模块的测试、验证或性能实验入口 |
| [pyproject.toml](repositories/context-language-models/pyproject.toml) | Python 包元数据、依赖与入口 |
| [suffix_cache_reuse/.gitignore](repositories/context-language-models/suffix_cache_reuse/.gitignore) | 仓库工具或版本控制配置 |
| [suffix_cache_reuse/README.md](repositories/context-language-models/suffix_cache_reuse/README.md) | 所在模块说明与使用方法 |
| [suffix_cache_reuse/assets/scr_bcp830_strip.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_bcp830_strip.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/assets/scr_full_vs_linear.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_full_vs_linear.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/assets/scr_ksweep.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_ksweep.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/assets/scr_overview.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_overview.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/assets/scr_prefill_split.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_prefill_split.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/assets/scr_strip_savings.png](repositories/context-language-models/suffix_cache_reuse/assets/scr_strip_savings.png) | 图形、论文插图或文档附件 |
| [suffix_cache_reuse/sglang/README.md](repositories/context-language-models/suffix_cache_reuse/sglang/README.md) | 所在模块说明与使用方法 |
| [suffix_cache_reuse/sglang/analysis/scr_report.py](repositories/context-language-models/suffix_cache_reuse/sglang/analysis/scr_report.py) | SCR 日志分析与报表 |
| [suffix_cache_reuse/sglang/examples/serve_qwen36.sh](repositories/context-language-models/suffix_cache_reuse/sglang/examples/serve_qwen36.sh) | 仓库辅助文件 |
| [suffix_cache_reuse/sglang/pyproject.toml](repositories/context-language-models/suffix_cache_reuse/sglang/pyproject.toml) | Python 包元数据、依赖与入口 |
| [suffix_cache_reuse/sglang/suffix_cache_reuse/__init__.py](repositories/context-language-models/suffix_cache_reuse/sglang/suffix_cache_reuse/__init__.py) | Python 包初始化与公开接口 |
| [suffix_cache_reuse/sglang/suffix_cache_reuse/_site/sitecustomize.py](repositories/context-language-models/suffix_cache_reuse/sglang/suffix_cache_reuse/_site/sitecustomize.py) | Python 启动时加载集成逻辑 |
| [suffix_cache_reuse/sglang/suffix_cache_reuse/config.py](repositories/context-language-models/suffix_cache_reuse/sglang/suffix_cache_reuse/config.py) | 该模块配置 |
| [suffix_cache_reuse/sglang/suffix_cache_reuse/overlay.py](repositories/context-language-models/suffix_cache_reuse/sglang/suffix_cache_reuse/overlay.py) | SGLang 运行时覆盖与接线 |
| [suffix_cache_reuse/sglang/suffix_cache_reuse/serve.py](repositories/context-language-models/suffix_cache_reuse/sglang/suffix_cache_reuse/serve.py) | 该服务启动入口 |
| [suffix_cache_reuse/sglang/tests/regression.py](repositories/context-language-models/suffix_cache_reuse/sglang/tests/regression.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/sglang/tests/smoke.py](repositories/context-language-models/suffix_cache_reuse/sglang/tests/smoke.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/sglang/tests/unit_test.py](repositories/context-language-models/suffix_cache_reuse/sglang/tests/unit_test.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/DESIGN.md](repositories/context-language-models/suffix_cache_reuse/vllm/DESIGN.md) | 设计文档 |
| [suffix_cache_reuse/vllm/README.md](repositories/context-language-models/suffix_cache_reuse/vllm/README.md) | 所在模块说明与使用方法 |
| [suffix_cache_reuse/vllm/analysis/scr_report.py](repositories/context-language-models/suffix_cache_reuse/vllm/analysis/scr_report.py) | SCR 日志分析与报表 |
| [suffix_cache_reuse/vllm/pyproject.toml](repositories/context-language-models/suffix_cache_reuse/vllm/pyproject.toml) | Python 包元数据、依赖与入口 |
| [suffix_cache_reuse/vllm/tests/bench.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/bench.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/bench_conc.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/bench_conc.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/exactness.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/exactness.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/fixtures/tiny-llama/config.json](repositories/context-language-models/suffix_cache_reuse/vllm/tests/fixtures/tiny-llama/config.json) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/greedy_server.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/greedy_server.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/regime_probe.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/regime_probe.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/sim.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/sim.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/smoke.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/smoke.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/test_sim.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/test_sim.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/tests/unit_test.py](repositories/context-language-models/suffix_cache_reuse/vllm/tests/unit_test.py) | 所在模块的测试、验证或性能实验入口 |
| [suffix_cache_reuse/vllm/vllm_scr/__init__.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/__init__.py) | Python 包初始化与公开接口 |
| [suffix_cache_reuse/vllm/vllm_scr/_site/sitecustomize.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/_site/sitecustomize.py) | Python 启动时加载集成逻辑 |
| [suffix_cache_reuse/vllm/vllm_scr/compat.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/compat.py) | 版本兼容逻辑 |
| [suffix_cache_reuse/vllm/vllm_scr/config.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/config.py) | 该模块配置 |
| [suffix_cache_reuse/vllm/vllm_scr/connector.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/connector.py) | vLLM 缓存连接 |
| [suffix_cache_reuse/vllm/vllm_scr/planner.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/planner.py) | 缓存复用规划 |
| [suffix_cache_reuse/vllm/vllm_scr/scheduler.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/scheduler.py) | vLLM 调度集成 |
| [suffix_cache_reuse/vllm/vllm_scr/serve.py](repositories/context-language-models/suffix_cache_reuse/vllm/vllm_scr/serve.py) | 该服务启动入口 |

## ContextBench

| 文件 | 用途 |
|---|---|
| [.gitignore](repositories/ContextBench/.gitignore) | 仓库工具或版本控制配置 |
| [LICENSE](repositories/ContextBench/LICENSE) | 许可证 |
| [README.md](repositories/ContextBench/README.md) | 所在模块说明与使用方法 |
| [clm_exp/README.md](repositories/ContextBench/clm_exp/README.md) | 所在模块说明与使用方法 |
| [clm_exp/configs/paper.yaml](repositories/ContextBench/clm_exp/configs/paper.yaml) | 该模块/任务的配置或元数据 |
| [clm_exp/skills/README.md](repositories/ContextBench/clm_exp/skills/README.md) | 所在模块说明与使用方法 |
| [clm_exp/skills/acm/kv_store.md](repositories/ContextBench/clm_exp/skills/acm/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/acm/log_triage.md](repositories/ContextBench/clm_exp/skills/acm/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/acm/needle_retention.md](repositories/ContextBench/clm_exp/skills/acm/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/acm/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/acm/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/kv_store.md](repositories/ContextBench/clm_exp/skills/base/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/kv_store.short.md](repositories/ContextBench/clm_exp/skills/base/kv_store.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/log_triage.md](repositories/ContextBench/clm_exp/skills/base/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/log_triage.short.md](repositories/ContextBench/clm_exp/skills/base/log_triage.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/needle_retention.md](repositories/ContextBench/clm_exp/skills/base/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/base/sudoku_sketchpad.short.md](repositories/ContextBench/clm_exp/skills/base/sudoku_sketchpad.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/clm/kv_store.md](repositories/ContextBench/clm_exp/skills/clm/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/clm/log_triage.md](repositories/ContextBench/clm_exp/skills/clm/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/clm/needle_retention.md](repositories/ContextBench/clm_exp/skills/clm/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/clm/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/clm/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/context_folding/kv_store.md](repositories/ContextBench/clm_exp/skills/context_folding/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/context_folding/log_triage.md](repositories/ContextBench/clm_exp/skills/context_folding/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/context_folding/needle_retention.md](repositories/ContextBench/clm_exp/skills/context_folding/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/context_folding/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/context_folding/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/rlm/kv_store.md](repositories/ContextBench/clm_exp/skills/rlm/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/rlm/log_triage.md](repositories/ContextBench/clm_exp/skills/rlm/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/rlm/needle_retention.md](repositories/ContextBench/clm_exp/skills/rlm/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/rlm/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/rlm/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/self_compact/kv_store.md](repositories/ContextBench/clm_exp/skills/self_compact/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/self_compact/log_triage.md](repositories/ContextBench/clm_exp/skills/self_compact/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/self_compact/needle_retention.md](repositories/ContextBench/clm_exp/skills/self_compact/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/self_compact/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/self_compact/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/kv_store.md](repositories/ContextBench/clm_exp/skills/summary/kv_store.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/kv_store.short.md](repositories/ContextBench/clm_exp/skills/summary/kv_store.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/log_triage.md](repositories/ContextBench/clm_exp/skills/summary/log_triage.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/log_triage.short.md](repositories/ContextBench/clm_exp/skills/summary/log_triage.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/needle_retention.md](repositories/ContextBench/clm_exp/skills/summary/needle_retention.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/needle_retention.short.md](repositories/ContextBench/clm_exp/skills/summary/needle_retention.short.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/sudoku_sketchpad.md](repositories/ContextBench/clm_exp/skills/summary/sudoku_sketchpad.md) | 对应方法与任务族的 Skill 正文 |
| [clm_exp/skills/summary/sudoku_sketchpad.short.md](repositories/ContextBench/clm_exp/skills/summary/sudoku_sketchpad.short.md) | 对应方法与任务族的 Skill 正文 |
| [contextbench/__init__.py](repositories/ContextBench/contextbench/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/cli.py](repositories/ContextBench/contextbench/cli.py) | 命令行接口 |
| [contextbench/configs/paper.yaml](repositories/ContextBench/contextbench/configs/paper.yaml) | 该模块/任务的配置或元数据 |
| [contextbench/feasibility.py](repositories/ContextBench/contextbench/feasibility.py) | 预算可行性检查 |
| [contextbench/graders/__init__.py](repositories/ContextBench/contextbench/graders/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/graders/base.py](repositories/ContextBench/contextbench/graders/base.py) | 模块说明 · Grader plumbing: the three ways a run's context is read back. |
| [contextbench/graders/kv.py](repositories/ContextBench/contextbench/graders/kv.py) | 模块说明 · Grader for *Key-Value Store*: process-graded, ANSWER-block based. |
| [contextbench/graders/log.py](repositories/ContextBench/contextbench/graders/log.py) | 模块说明 · Grader for *Incident Log Triage*: process-graded, ANSWER-block based. |
| [contextbench/graders/needle.py](repositories/ContextBench/contextbench/graders/needle.py) | 模块说明 · Grader for *Needle Retention*: what is still in the context at the end. |
| [contextbench/graders/sudoku.py](repositories/ContextBench/contextbench/graders/sudoku.py) | 模块说明 · Grader for *Sudoku Sketchpad*: versioned, process-graded. |
| [contextbench/harbor/__init__.py](repositories/ContextBench/contextbench/harbor/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/harbor/_mswea_driver.py](repositories/ContextBench/contextbench/harbor/_mswea_driver.py) | 模块说明 · In-container driver for ``MiniSweAgentStream``: one mini-swe-agent step per call. |
| [contextbench/harbor/agents/__init__.py](repositories/ContextBench/contextbench/harbor/agents/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/harbor/agents/base.py](repositories/ContextBench/contextbench/harbor/agents/base.py) | 模块说明 · ``ContextBenchAgent``: a minimal bash-tool agent loop for Harbor. |
| [contextbench/harbor/agents/terminus2_stream.py](repositories/ContextBench/contextbench/harbor/agents/terminus2_stream.py) | 模块说明 · ``Terminus2Stream``: Harbor's own terminus-2 agent, fed by a ContextBench stream. |
| [contextbench/harbor/mixin.py](repositories/ContextBench/contextbench/harbor/mixin.py) | 为 agent 接入流式基准的 mixin |
| [contextbench/harbor/package.py](repositories/ContextBench/contextbench/harbor/package.py) | Harbor 任务打包 |
| [contextbench/harbor/reward.py](repositories/ContextBench/contextbench/harbor/reward.py) | 奖励结果处理 |
| [contextbench/harbor/turn_source.py](repositories/ContextBench/contextbench/harbor/turn_source.py) | 下一操作输入来源 |
| [contextbench/harbor/verifiers.py](repositories/ContextBench/contextbench/harbor/verifiers.py) | Harbor 评分适配 |
| [contextbench/harbor/wrap.py](repositories/ContextBench/contextbench/harbor/wrap.py) | 模块说明 · ``cb_stream``: run a Harbor installed CLI agent on a ContextBench stream. |
| [contextbench/hf_export.py](repositories/ContextBench/contextbench/hf_export.py) | Hugging Face 数据导出 |
| [contextbench/levels.py](repositories/ContextBench/contextbench/levels.py) | 模块说明 · Levels of a levels file (``clm_exp/configs/paper.yaml``): generator knobs vs descriptive keys. |
| [contextbench/oracle.py](repositories/ContextBench/contextbench/oracle.py) | 参考处理逻辑 |
| [contextbench/runner.py](repositories/ContextBench/contextbench/runner.py) | 基准运行入口 |
| [contextbench/stream.py](repositories/ContextBench/contextbench/stream.py) | 操作流、READY 与终止协议 |
| [contextbench/tasks/__init__.py](repositories/ContextBench/contextbench/tasks/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/tasks/kv_store/__init__.py](repositories/ContextBench/contextbench/tasks/kv_store/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/tasks/kv_store/generate.py](repositories/ContextBench/contextbench/tasks/kv_store/generate.py) | 当前任务族实例生成 |
| [contextbench/tasks/kv_store/scaffold.md](repositories/ContextBench/contextbench/tasks/kv_store/scaffold.md) | 当前任务族指令模板 |
| [contextbench/tasks/log_triage/__init__.py](repositories/ContextBench/contextbench/tasks/log_triage/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/tasks/log_triage/generate.py](repositories/ContextBench/contextbench/tasks/log_triage/generate.py) | 当前任务族实例生成 |
| [contextbench/tasks/log_triage/scaffold.md](repositories/ContextBench/contextbench/tasks/log_triage/scaffold.md) | 当前任务族指令模板 |
| [contextbench/tasks/needle_retention/__init__.py](repositories/ContextBench/contextbench/tasks/needle_retention/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/tasks/needle_retention/generate.py](repositories/ContextBench/contextbench/tasks/needle_retention/generate.py) | 当前任务族实例生成 |
| [contextbench/tasks/needle_retention/scaffold.md](repositories/ContextBench/contextbench/tasks/needle_retention/scaffold.md) | 当前任务族指令模板 |
| [contextbench/tasks/scaffold.py](repositories/ContextBench/contextbench/tasks/scaffold.py) | 模块说明 · Instruction scaffolds: the family contract text handed to the agent. |
| [contextbench/tasks/sudoku_sketchpad/__init__.py](repositories/ContextBench/contextbench/tasks/sudoku_sketchpad/__init__.py) | Python 包初始化与公开接口 |
| [contextbench/tasks/sudoku_sketchpad/generate.py](repositories/ContextBench/contextbench/tasks/sudoku_sketchpad/generate.py) | 当前任务族实例生成 |
| [contextbench/tasks/sudoku_sketchpad/scaffold.md](repositories/ContextBench/contextbench/tasks/sudoku_sketchpad/scaffold.md) | 当前任务族指令模板 |
| [contextbench/tokens.py](repositories/ContextBench/contextbench/tokens.py) | token 统计和快照工具 |
| [docs/DESIGN.md](repositories/ContextBench/docs/DESIGN.md) | 设计文档 |
| [docs/DEVIATIONS.md](repositories/ContextBench/docs/DEVIATIONS.md) | 该主题的文档，文件名标识具体主题 |
| [docs/HARBOR_SMOKE.md](repositories/ContextBench/docs/HARBOR_SMOKE.md) | 该主题的文档，文件名标识具体主题 |
| [environment/contextbench-cli-agents.def](repositories/ContextBench/environment/contextbench-cli-agents.def) | 仓库辅助文件 |
| [environment/contextbench.def](repositories/ContextBench/environment/contextbench.def) | 仓库辅助文件 |
| [patches/harbor-0.16-terminus2-turn-source.patch](repositories/ContextBench/patches/harbor-0.16-terminus2-turn-source.patch) | 外部训练框架的版本化代码补丁 |
| [pyproject.toml](repositories/ContextBench/pyproject.toml) | Python 包元数据、依赖与入口 |
| [registry.json](repositories/ContextBench/registry.json) | 该模块/任务的配置或元数据 |
| [tests/conftest.py](repositories/ContextBench/tests/conftest.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_cli.py](repositories/ContextBench/tests/test_cli.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_feasibility.py](repositories/ContextBench/tests/test_feasibility.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_generators.py](repositories/ContextBench/tests/test_generators.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_graders.py](repositories/ContextBench/tests/test_graders.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_harbor_extras.py](repositories/ContextBench/tests/test_harbor_extras.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_harbor_package.py](repositories/ContextBench/tests/test_harbor_package.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_harbor_turn_source.py](repositories/ContextBench/tests/test_harbor_turn_source.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_hf_export.py](repositories/ContextBench/tests/test_hf_export.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_install.py](repositories/ContextBench/tests/test_install.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_runner.py](repositories/ContextBench/tests/test_runner.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_skills.py](repositories/ContextBench/tests/test_skills.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_stream.py](repositories/ContextBench/tests/test_stream.py) | 所在模块的测试、验证或性能实验入口 |
| [tests/test_tokens.py](repositories/ContextBench/tests/test_tokens.py) | 所在模块的测试、验证或性能实验入口 |
| [tools/audit_trial.py](repositories/ContextBench/tools/audit_trial.py) | 单题输入流与轨迹审计 |

## 每个生成任务内部的文件

以下以 `kv-store-R1000-seed42` 为例，各实例数值和输入不同。

| 相对任务目录的路径 | 用途 |
|---|---|
| `environment/stream/chunks.jsonl` | 按顺序推送给模型的操作流 |
| `instruction.md` | 执行器任务说明 |
| `solution/solve.sh` | 参考解法文件 |
| `task.toml` | 预算、资源、超时、镜像和输入流哈希 |
| `tests/_cb/contextbench/__init__.py` | 评分侧脚本或其依赖 · Python 包初始化与公开接口 |
| `tests/_cb/contextbench/graders/__init__.py` | 评分侧脚本或其依赖 · Python 包初始化与公开接口 |
| `tests/_cb/contextbench/graders/base.py` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/_cb/contextbench/graders/kv.py` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/_cb/contextbench/graders/log.py` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/_cb/contextbench/graders/needle.py` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/_cb/contextbench/graders/sudoku.py` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/_cb/contextbench/harbor/__init__.py` | 评分侧脚本或其依赖 · Python 包初始化与公开接口 |
| `tests/_cb/contextbench/harbor/mixin.py` | 评分侧脚本或其依赖 · 为 agent 接入流式基准的 mixin |
| `tests/_cb/contextbench/harbor/reward.py` | 评分侧脚本或其依赖 · 奖励结果处理 |
| `tests/_cb/contextbench/harbor/turn_source.py` | 评分侧脚本或其依赖 · 下一操作输入来源 |
| `tests/_cb/contextbench/runner.py` | 评分侧脚本或其依赖 · 基准运行入口 |
| `tests/_cb/contextbench/stream.py` | 评分侧脚本或其依赖 · 操作流、READY 与终止协议 |
| `tests/_cb/contextbench/tokens.py` | 评分侧脚本或其依赖 · token 统计和快照工具 |
| `tests/feasibility_receipt.json` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
| `tests/gold.json` | 评分侧正确答案 |
| `tests/meta.json` | 实例生成参数与统计 |
| `tests/test.sh` | 评分侧脚本或其依赖 · 所在模块的测试、验证或性能实验入口 |
