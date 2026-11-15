# Context Language Models 论文资源清单

核查日期为 2026-11-14。对应论文为 Rulin Shao 等人的《Context Language Models》，arXiv 2609.37725v1。此次检查了论文正文与附录、TeX 源码、官方代码文件树、ContextBench 仓库、Hugging Face 模型与数据文件，以及官方链接的训练和推理依赖。

本清单现附有第二轮扩展调查，新增 16 个入口，包含 Tiny CLM 的 RL 复现、94 份 LoRA、3,750 条轨迹记录，以及 Hermes 340 轮对照与公开数据。详细核验和局限见 [社区复现与扩展资源](COMMUNITY_REPRODUCTIONS.md)，索引见 [新增资源 CSV](EXPANDED_RESOURCES.csv)。

## 1. 已找到的资源

| 资源 | 实际内容 | 获取情况 |
|---|---|---|
| 论文与附录 | HTML、PDF、TeX、图表和表格源码 | 可获取，TeX 已下载 |
| CLM 主代码 | Harness、上下文编辑、Skill 演化程序、RL 补丁、缓存复用实现 | 已克隆并检查关键实现 |
| ContextBench 方法 Skill 库 | 7 种方法、34 份 Markdown，其中 CLM 方法 4 份 | 全部已读取并建立 SHA256 索引 |
| RL 模型 | LiveCtx iter44 和 b0sum iter49，各约 19.306 GB 权重 | 文件元数据与匿名下载入口通过检查 |
| ContextBench | 4 类、256 个独立实例，含输入流、gold、生成器和评分器 | 仓库已下载，HF 数据入口可获取 |
| 相关压缩轨迹 | 两份公开数据合计 68,045 行分段样本 | 已检查字段和完整单行样本，论文使用关系待确认 |
| 相关 SFT 轨迹 | 42 个 SWE 任务，包含 234 次 edit_context 调用 | JSONL 已完整下载并计数，论文使用关系待确认 |
| 轨迹与计算量工具 | ATIF-CTX 格式、上下文快照、FLOPs 统计、审计脚本 | 实现可获取 |
| 插件和解说 | Pi 插件、OpenCode 社区插件、作者 SCR 博客 | 入口与说明已核查 |

已定位的缺口包括完整 ICL 演化候选与最终 Skill 档案、论文正式实验的全量请求和响应、精确 RL 数据划分，以及部分原始训练 checkpoint 文件。这些缺口的具体证据见后文。

## 2. 论文和官方入口

| 入口 | 内容与用途 |
|---|---|
| [论文页面](https://arxiv.org/abs/2609.37725) | 摘要、作者、版本记录，目前检查到 v1 |
| [HTML 正文与附录](https://arxiv.org/html/2609.37725v1) | 检索方法、配置、实验和 Skill 示例 |
| [PDF](https://arxiv.org/pdf/2609.37725v1) | 完整排版论文 |
| [TeX 源码](https://arxiv.org/src/2609.37725v1) | 已下载约 3.14 MB，解包得到 49 个文件，包含论文、附录、参考文献、图表与表格源码 |
| [官方主仓库](https://github.com/facebookresearch/context-language-models) | CLM 运行、演化、RL 和 SCR 实现 |
| [Hugging Face 组织](https://huggingface.co/context-language-model) | 本次列出 2 个模型仓库和 1 个数据集 |
| [ContextBench 仓库](https://github.com/RulinShao/ContextBench) | 任务、方法 Skill、生成器、grader 和接线审计 |
| [作者 SCR 博客](https://wai-org.com/blog/clm/) | 缓存复用与 FLOPs 的交互解说 |

TeX 包提供结果表格和图形源文件，当前包内未找到全量实验 CSV/JSON 或 rollout。主仓库 Releases 接口当前返回空列表。

## 3. Skill 仓库和具体内容

### 3.1 可以直接取得的 34 份方法 Skill

文件位于 ContextBench 的 `clm_exp/skills/`。该目录 README 记录方法来源、文件哈希及论文 ContextBench 图中的使用关系。全部文件已保存到本清单的 `evidence/ContextBench/`，逐文件链接、大小和 SHA256 见 [SKILL_FILES.csv](SKILL_FILES.csv)。[固定版本的 Skill 目录](https://github.com/RulinShao/ContextBench/tree/fb533c18c78a82d3f1b2047a9c1700735cf829ae/clm_exp/skills)

| 方法目录 | 文件数 | 内容 |
|---|---:|---|
| `clm` | 4 | 通过文件存储和编辑当前上下文管理历史 |
| `acm` | 4 | manage_context、query_memory 等操作说明 |
| `rlm` | 4 | 对应 RLM 接口的上下文处理说明 |
| `context_folding` | 4 | 折叠上下文的方法说明 |
| `self_compact` | 4 | 自主压缩上下文的方法说明 |
| `summary` | 8 | 4 份完整版本和 4 份短版本 |
| `base` | 6 | 3 份完整版本和 3 份短版本 |
| 合计 | **34** | 27 份完整版本和 7 份短版本 |

其中 4 份 CLM Skill 如下。已计算的 SHA256 前 16 位与主仓库适配器固定值相同。

| 文件 | 实际指导内容 | SHA256 前 16 位 |
|---|---|---|
| [kv_store.md](evidence/ContextBench/clm_exp/skills/clm/kv_store.md) | 将 SET 批次写入文件，保留定位标记，清理旧命令轨迹，需要时从文件找回值 | `70695bdca1a7f39d` |
| [log_triage.md](evidence/ContextBench/clm_exp/skills/clm/log_triage.md) | 保存日志批次，查询已保存日志，控制命令记录带来的上下文增长 | `1a071e820bcc9a93` |
| [needle_retention.md](evidence/ContextBench/clm_exp/skills/clm/needle_retention.md) | 保留 needle 的准确文本，删除填充内容，清理冗余操作记录 | `e834d24bf004c7ae` |
| [sudoku_sketchpad.md](evidence/ContextBench/clm_exp/skills/clm/sudoku_sketchpad.md) | 原地更新棋盘和版本，检查棋盘结构，删除旧移动记录 | `378c4e5d928af06f` |

这批文件有明确的方法和任务族划分。论文 ICL 阶段演化出的最终 Skill、各轮候选和选择记录，需要另行获取完整运行档案。

### 3.2 Skill 的生成、优化与加载代码

| 位置 | 已发布的能力 |
|---|---|
| `clm/clm_icl/` | 运行训练轨迹，分析候选，生成新的完整 Skill，验证格式与证据引用，执行 dev 评估和接受门，记录 Pareto 结果 |
| `clm/clm_icl/proposer_prompt.md` | Skill proposer 的提示词，要求引用轨迹步骤并说明预期影响 |
| `clm/clm_harness/utils/skills.py` | 上传 Skill 目录，配置 PATH 和声明的环境变量，将 SKILL.md 加入提示 |

对应源码入口分别见 [ICL](https://github.com/facebookresearch/context-language-models/tree/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_icl)、[proposer 提示词](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_icl/proposer_prompt.md) 和 [Skill 加载器](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/utils/skills.py)。

演化程序约定生成以下产物。

```text
state.json
skills/<id>/SKILL.md
rounds/r<k>/note.md
rounds/r<k>/proposer_request.txt
rounds/r<k>/proposer_response.txt
rounds/r<k>/analysis.md
rounds/r<k>/gate.json
episodes/...
trials/...
best_skill/SKILL.md
```

当前公开主仓库中未找到论文完整运行对应的上述目录和最终 `best_skill/SKILL.md`。论文 BrowseComp-Plus 实验提到的检索工具 Skill，其挂载机制公开，原始 `deep_research`、`web_search` Skill 包尚未定位。

## 4. 训练后模型的存放位置

### 4.1 两个公开 checkpoint

| 项目 | LiveCtx / CLM | b0sum 对照 checkpoint |
|---|---|---|
| HF 仓库 | `context-language-model/rl-ckpt-livectx_dapo-iter44` | `context-language-model/rl-ckpt-b0sum_dapo-iter49` |
| 可加载文件位置 | **`converted/` 子目录** | **仓库根目录** |
| 文件 | config、tokenizer、index、4 个 safetensors 分片 | config、tokenizer、index、4 个 safetensors 分片 |
| 权重合计 | 19,306,303,904 B，约 19.306 GB | 19,306,303,904 B，约 19.306 GB |
| 配置 | Qwen3.5 系列 9B，架构字段 Qwen3_5ForConditionalGeneration，文本 dtype 为 bfloat16 | 同系列、同架构和 dtype 配置 |
| 下载状态 | public、nongated，4 个权重链接匿名 HEAD 200 | public、nongated，4 个权重链接匿名 HEAD 200 |
| 描述充分程度 | README 明确说明 LiveCtx arm checkpoint | 通用 Qwen 模型卡，缺少 checkpoint 专门训练说明 |

模型入口见 [LiveCtx converted 文件](https://huggingface.co/context-language-model/rl-ckpt-livectx_dapo-iter44/tree/main/converted) 和 [b0sum 根目录](https://huggingface.co/context-language-model/rl-ckpt-b0sum_dapo-iter49/tree/main)。b0sum 的 summary 对照角色依据仓库命名判断，精确训练配置及论文表格行映射仍缺少专门说明。

两组权重的分片大小相同，LFS SHA256 不同。本次核对了模型配置、索引、MANIFEST、文件大小、LFS 对象哈希和下载状态，尚未下载约 38.61 GB 的全部权重进行二进制哈希或推理验证。逐分片记录见 [MODEL_FILES.csv](MODEL_FILES.csv)，下载检查见 [download-checks.json](evidence/download-checks.json)。

### 4.2 raw checkpoint 的实际缺口

LiveCtx README 声明包含 `raw/iter_0000044`。其 MANIFEST 列出 35 个 raw 文件，合计 **125,359,119,492 B，约 125.359 GB**，包括 32 个 `.distcp` 分片及训练元数据文件。

当前公开文件树未出现 `raw/`。抽查 `raw/iter_0000044/metadata.json` 和 `raw/iter_0000044/__0_0.distcp` 均返回 HTTP 404。现阶段可以获取导出的模型权重与 raw 文件清单，原始训练状态的获取仍有缺口。这里的 raw 指分布式训练 checkpoint，分析训练轨迹还需要对应的请求、响应与奖励日志。[模型 MANIFEST](https://huggingface.co/context-language-model/rl-ckpt-livectx_dapo-iter44/blob/main/MANIFEST.tsv)

## 5. 数据、原始轨迹和 Skill 编辑信息

### 5.1 ContextBench 正式配套资源

| 任务族 | 独立实例数 | 主要数据 |
|---|---:|---|
| kv_store | 64 | 键值写入、更新和后续查询 |
| log_triage | 56 | 日志流及诊断查询 |
| needle_retention | 80 | 大量填充文本中的准确事实保留 |
| sudoku_sketchpad | 56 | 棋盘状态和移动记录 |
| 合计 | **256** | 32 个 level × 8 个 seed |

Hugging Face 上 4 个 Parquet 共 **58,006,858 B**。字段包含 `task_id`、`family`、`level`、`seed`、`budget`、`reserve`、`n_ops`、`streamed_tokens`、`pressure`、`chunks`、`gold`、`meta`。页面/API 汇总的 512 行包含 `all` 配置的 256 行及四个子配置的同批 256 行，独立实例数为 256。[HF 数据集](https://huggingface.co/datasets/context-language-model/ContextBench)

GitHub 还提供 `chunks.jsonl`、`gold.json`、`meta.json`、任务指令、task.toml、生成器、解法、测试与评分代码，以及 Docker/Apptainer 环境。`docs/DEVIATIONS.md` 记录公开移植版的 tokenizer、指令、评分来源等差异，复现时应按该文档固定条件。[ContextBench 仓库](https://github.com/RulinShao/ContextBench)、[差异记录](https://github.com/RulinShao/ContextBench/blob/master/docs/DEVIATIONS.md)

### 5.2 已找到的相关原始数据

以下资源可以提供上下文编辑样本。其与论文正式训练或评测运行的精确对应关系，当前公开材料尚未确认。

| 数据集 | 规模 | 已检查的内容 | 可获取状态 |
|---|---:|---|---|
| `osieosie/drctx-compaction-openresearcher-mk1` | 38,446 行，约 3.788 GB | messages、tools、source、metadata，含 reasoning_content 和工具调用 | size/schema/完整单行样本已核 |
| `osieosie/openresearcher-compaction-mk2-nc1` | 29,599 行，约 2.609 GB | 同类消息和压缩分段结构 | size/schema/完整单行样本已核 |
| `rulins/memory_sft_data` | 42 个 instance_id，约 2.40 MB | 42 行完整轨迹、234 次 edit_context 调用 | train.jsonl 已完整下载 |
| `OpenResearcher/OpenResearcher-Dataset` | 上游卡片列约 97,630 条轨迹，约 7.46 GB | 原始研究 agent 数据来源 | 数据卡和上游代码说明已核 |
| `wAI-org/OpenResearcher-Dataset` | 卡片列 sft/rl 各 3,051 题 | qid、question、answer | RL Parquet 已下载，逐行数尚未独立解析 |

对应入口见 [mk1](https://huggingface.co/datasets/osieosie/drctx-compaction-openresearcher-mk1)、[mk2](https://huggingface.co/datasets/osieosie/openresearcher-compaction-mk2-nc1)、[memory_sft_data](https://huggingface.co/datasets/rulins/memory_sft_data)、[OpenResearcher 上游](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) 和 [WAI 问答数据](https://huggingface.co/datasets/wAI-org/OpenResearcher-Dataset)。

两份 compaction 数据合计 **68,045 行分段样本**。metadata 包含 `session_id`、`trajectory_id`、`segment_index`、`n_segments`、`compaction_effort`、`has_compaction`、`session_self_compacted` 和 `edit_label`。编辑标签包括类型、跨轮标志、目标年龄、是否应用、删除字符数和 offload 文件名等，可以关联编辑行为与后续上下文。行数统计需要结合 session/trajectory 去重后才能得到独立任务量。

mk1 抽查样本有 42 条 messages，原会话分为 4 段，含 bash、edit_context 与应用成功的 full_compaction 标签。mk2 抽查样本有 10 条 messages、1 段。两者当前数据卡主要为 YAML 元数据，未找到明确论文引用和许可证声明，使用前需要继续核对授权条件。

`memory_sft_data` 已逐行检查，42 个独立 instance_id，每条 34–78 条消息，工具包含 bash 和 edit_context，调用计数与 `n_edit_turns` 字段合计一致。数据卡声明来自 SWE-bench Verified、使用 GPT-5.4 oracle-guided teacher，并标注 ODC-By。声明中的 oracle-blind rationale 尚未在本次进行泄漏评估。

论文使用的 **3,040 个 RL 训练样本和 500 个验证样本**，当前未找到可以逐项映射的公开划分清单。WAI 镜像的 3,051 题需要进一步确认筛选过程，且其声明的 eval 文件模式在当前文件树缺少匹配文件。

### 5.3 论文正式 rollout 的发布程度

| 产物 | 当前核查结果 |
|---|---|
| 论文全部正式运行的模型请求、响应、工具返回 | 未在官方主仓库、ContextBench、HF 组织和 TeX 包中定位完整档案 |
| ICL 全候选、每轮 proposer 输入输出、gate 和最终 Skill | 程序的生成路径公开，论文完整产物未定位 |
| ContextBench 工程运行样例 | `docs/HARBOR_SMOKE.md` 有 18 个 trial 小节，提供分数、快照审计、取消或失败记录与节选 |
| 逐次上下文变化格式和校验代码 | ATIF-CTX、快照及审计器可获取 |
| BrowseComp-Plus baseline 原始运行 | 上游仓库提供下载脚本，归属 BCP 作者的 baseline |
| SCR 原始服务日志和曲线数据 | 分析程序与部分图可获取，完整原始实验日志未定位 |

相关入口见 [Harbor smoke 记录](https://github.com/RulinShao/ContextBench/blob/master/docs/HARBOR_SMOKE.md)、[audit_trial.py](https://github.com/RulinShao/ContextBench/blob/master/tools/audit_trial.py) 和 [BCP 上游运行下载脚本](https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_build_index/download_run_files.sh)。

## 6. 运行、训练和分析工具

| 模块 | 位置与内容 | 获取后需要注意的条件 |
|---|---|---|
| Harness | `clm/clm_harness`，agent、上下文文件、预算、编辑反馈、checkpoint/resume | Python >=3.12,<3.14，Harbor 0.16.1 等依赖 |
| 轨迹格式 | `agent_trajectory_format`，input_context、segments、branch、subtrajectories、SpawnLink 与校验器 | 可以表达上下文被编辑后的真实模型输入 |
| 运行留档 | trajectory.json、trajectory.ctx.json、usage.json、timing.json、context_snapshots/turn-NNNN.json | 需要实际运行生成，目录契约已公开 |
| ContextBench adapter | 流式任务适配器，增加 interactive_stream.json 和 clm_stream.json | 结合输入流和快照核对上下文操作 |
| 计算量 | `flops_metrics/kv_cache_flops.py` | 根据上下文轨迹计算 prefix reuse 下的 prefill/decode FLOPs |
| RL 改动 | `clm/clm_rl` README 和两个 patch | 对接 ProRL-Agent-Server 与 Slime |
| SGLang SCR | `suffix_cache_reuse/sglang` | 面向 SGLang 0.5.16，有启动、测试和日志分析说明 |
| vLLM SCR | `suffix_cache_reuse/vllm` | 面向 vLLM 0.19.1，有设计、测试和 benchmark 程序 |

入口见 [Harness](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_harness)、[RL](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_rl) 和 [SCR](https://github.com/facebookresearch/context-language-models/tree/main/suffix_cache_reuse)。

RL 补丁分别指向 ProRL commit `8bc67cc` 和 Slime commit `bf9b1a3`，公开双通道优势的关键改动。补丁消费 `sample.metadata["a_eff"]` 和 `sample.metadata["polar"]["role_mask"]`，其奖励后处理和角色标记生产逻辑仍需结合外部系统实现。仓库目前缺少可以直接还原论文全程训练的完整 launch/config/log 包。[ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)、[Slime](https://github.com/THUDM/slime)

论文中的权重学习、ICL Skill 文本优化与零样本上下文编辑分别有不同产物。两个 HF checkpoint 对应权重产物，34 份方法 Skill 对应配套指令，相关压缩数据提供消息和编辑标签。复现时应分别记录其来源版本。

## 7. 其他评测与集成资源

| 资源 | 公开内容与论文关系 |
|---|---|
| [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | 830 个检索问题，搜索与评分实现 |
| [BCP HF collection](https://huggingface.co/collections/Tevatron/browsecomp-plus) | 问题、约 10 万文档语料和预构建索引 |
| [EdgeBench](https://github.com/ByteDance-Seed/EdgeBench) | 环境与评分代码，CLM 使用其中 10 个任务，复现需固定原版本 |
| [Terminal-Bench 2](https://github.com/harbor-framework/terminal-bench-2) | CLM 指定 2.1 任务版本，当前仓库入口可用 |
| [OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite) | 100 个终端任务 |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | 数学优化对照实现，论文讨论四类优化问题 |
| Software World | 附录描述六仓库协同任务和下游评测，完整对应任务与轨迹包尚未定位 |
| [Pi CLM](https://github.com/lolipopshock/pi-clm) | 主仓库链接的集成，可看上下文、input、edit diff，npm 包 @lolipopshock/pi-clm，本次版本 1.0.0 |
| [OpenCode CLM](https://github.com/bcmyguest/opencode-clm) | 社区移植，提供插件与上下文面板，主仓库标明由社区维护 |

论文 EdgeBench 十题名称为 ad_placement_optimization、apple_incremental_game、graph_node_classification、grid_turing_robot、juliet_vulnerability_analyzer、openrct2_theme_park_ai、schemathesis_datagen_pipeline、triangulation_coloring_optimization、vehicle_routing_time_windows 和 wesnoth_tactical_ai。

主仓库当前开放的 PR13 涉及上下文编辑后保留未编辑消息中的结构化工具调用与 reasoning 元数据。本次检查时尚未合入所查 main，后续复现可核对其合并状态。[PR13](https://github.com/facebookresearch/context-language-models/pull/13)

## 8. 本地资源和核验记录

本次已整理 **44 项资源索引**、34 份 Skill 文件索引和 8 个模型分片索引。代码、任务、Skill、小体积元数据与抽样数据保存在以下目录，完整模型和两份 GB 级 compaction 数据尚未下载。

```text
reports/clm-resources-20261114/
├── RESOURCE_CATALOG.md
├── RESOURCE_CATALOG.csv
├── RESOURCE_CATALOG.json
├── SKILL_FILES.csv
├── MODEL_FILES.csv
├── PROVENANCE.json
└── evidence/
    ├── context-language-models/
    ├── ContextBench/
    ├── paper-source/
    ├── paper-source.tar
    ├── download-checks.json
    └── 其他模型与数据 API、配置、清单、样本和哈希记录
```

| 固定对象 | 本次 revision |
|---|---|
| CLM 主仓库 | `483198229ab7b90c0eed4651306ab3b3505aaf50` |
| ContextBench 仓库 | `fb533c18c78a82d3f1b2047a9c1700735cf829ae` |
| LiveCtx HF | `ffc4aab2f30cb5b802ced6c84e6590ab5fec213d` |
| b0sum HF | `4a35d932e1112a2a9d98e9ca41f3c2c5f0cc2284` |
| ContextBench HF | `52bd961207ae9a0a79631c6fb65b162af9272983` |

许可证需要按资源分别核对。CLM 主仓库为 CC BY-NC 4.0，ContextBench 为 Apache 2.0，memory_sft_data 标注 ODC-By，其他数据和模型按各自文件声明处理。

机器可读清单见 [CSV](RESOURCE_CATALOG.csv) 和 [JSON](RESOURCE_CATALOG.json)。来源提交、元数据哈希和核验范围见 [PROVENANCE.json](PROVENANCE.json)。文件列表和 HEAD 检查能够确认入口及声明，模型加载、训练可复现性和 GB 级数据的逐行内容仍需运行验证。
