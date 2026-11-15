# CLM 扩展资源与社区复现调查

核查日期为 2026-11-14。本轮在原官方资源清单基础上扩大英文、中文、GitHub、GitLab、Hugging Face 与社区讨论检索，找到两个附有实际实验资料的项目。当前找到的实验分别采用小模型合成任务和 Hermes 长会话，原论文全部任务与 RL 数值的完整独立复现仍未定位。

## 1. Tiny CLM 小规模 RL 复现

作者为 Sugi Venugeethan。项目提供预注册假设、实验配置、代码、结果、偏离原论文的说明、训练日志、轨迹与模型适配器。代码已完整克隆到 `evidence/expanded/tiny-clm/`，提交为 `52a34a8120a7728cfaec523f48c4aa23af6a3f6d`。[代码仓库](https://gitlab.com/sugix/tiny-clm)

| 资源 | 内容和实际检查 |
|---|---|
| `PREREGISTRATION.md` | 假设、指标、判定规则及修订记录 |
| `RESULTS.md` | 分条件与分种子结果，失败记录，配置偏离与统计局限 |
| `env/kvstore.py`、`env/actions.py` | 2,048 token 预算的合成 KV 任务，OFFLOAD/GREP/NOTE/ANSWER/READY 操作 |
| `rl/advantages.py`、`rl/grpo_stepwise.py` | 结果奖励、成功条件下的效率优势、直接成本惩罚，逐调用训练实现 |
| `results/*/train_log.jsonl` | 各训练运行的指标日志 |
| `results/*/test/episodes.jsonl` | 逐题准确率、溢出、FLOPs、调用数和动作计数 |
| `results/**/trajectories.jsonl.gz` | **15 个文件、3,750 条轨迹记录、50,652 条 calls**，本次实际解包计数 |
| `blog_export/sample_trajectories/` | 5 份便于阅读的示例，对应 SFT、R1、R2、R3、final-transcript |
| `blog_export/*.json` | 图表数据、训练曲线与汇总表 |
| `results/weights_sha256.json` | 权重清单，可与 HF LFS 对象哈希匹配 |

轨迹里的每次调用包含 `context`、`output`、`prompt_tokens`、`gen_tokens`、`reused_tokens`、`flops`、`edit_depth` 和 `op`。已抽看真实输入、输出与动作。3,750 是多个实验条件、种子下的轨迹记录数，同一评测实例会被重复使用。它们可用于检查编辑前的模型输入及编辑决策，公开字段没有完整 HTTP/SSE 包和逐 token 时间戳。对应文件路径见 [本机仓库](evidence/expanded/tiny-clm/)，独立计数见 [AUDIT.json](evidence/expanded/AUDIT.json)。

### 训练后的模型

HF 仓库为 `sugiv/tiny-clm-qwen3-1.7b-lora`，基础模型为 Qwen/Qwen3-1.7B。文件树实际列出 **94 个 adapter_model.safetensors，共 13,114,219,744 B，约 13.114 GB**，其中包含保存的参考适配器副本。94 个文件的 LFS SHA256 全部与代码仓库清单一致，权重二进制尚未下载验证。HF revision 为 `0016ac03e5140b6e0b4abddeb81d55684a5a5bcf`。[模型仓库](https://huggingface.co/sugiv/tiny-clm-qwen3-1.7b-lora)

| 子目录 | 用途 |
|---|---|
| `e2_cal/adapter_step8/` | RL 使用的 SFT warm start |
| `e3_r1_s{0,1,2}/ckpt_best/` | 仅结果奖励，三个训练种子 |
| `e3_r2_s{0,1,2}/ckpt_best/` | 结果奖励加成功门控效率优势，w_eff=0.25 |
| `e3_r3_s{0,1,2}/ckpt_best/` | 直接成本惩罚 |
| `e5_final_s{0,1,2}/ckpt_best/` | final-transcript 训练消融 |
| `pilot_s0/`、`pilot2_s0/` | 学习率试跑与失败保留 |

### 结果及解释

本次从逐题 JSONL 独立重算了三种 RL 条件在压力 ≥2× 下的平均准确率，结果与仓库表格相符。

| 条件 | 三种子汇总准确率 | 单种子准确率 |
|---|---:|---|
| R1，仅结果奖励 | 85.39% | 100.00%、56.33%、99.83% |
| R2，成功门控效率优势 | 70.78% | 56.83%、55.67%、99.83% |
| R3，直接成本惩罚 | 85.33% | 57.00%、99.00%、100.00% |

该实验支持小模型通过 RL 学会上下文管理。各奖励条件只有三个训练种子，成功跳出 SFT 平台的数量不同，作者明确指出按评测 episode bootstrap 得到的区间低估了训练种子不确定性。当前数据不足以判定效率奖励优劣。合成任务中“立即 offload”同时有利于准确率和缓存，因此仅结果奖励也能学到尾部编辑策略。[结果与局限](https://gitlab.com/sugix/tiny-clm/-/blob/main/RESULTS.md)

其配置与论文有明显差异，包括 1.7B + LoRA、受限动作、合成 KV 环境、4 prompts × 8 rollouts、40 步训练。实验作者公布约 9 GPU-hours、约 4.65 美元计算成本，费用是作者自报。伴随博客可以帮助理解实验过程，模型和轨迹入口已从仓库另行核对。[复现实验文章](https://blog.sugiv.fyi/tiny-clm-teaching-a-tiny-model-to-manage-its-own-memory)

## 2. Hermes CLM + Context Map 长会话实验

`John-MiracleWorker/hermes-clm-showcase` 提供一套受 CLM 启发的 Hermes 上下文管理实现，使用 compact、archive、lookup、restore 等工具及 Context Map。仓库包含实现、插件、测试、任务生成器、报告和运行数据，许可证为 MIT。[仓库](https://github.com/John-MiracleWorker/hermes-clm-showcase)

| 资源 | 实际内容 |
|---|---|
| `benchmarks/reports/2026-10-14-long-horizon-breakdown.md` | 340 轮完整对照、故障与修复过程 |
| `data/results-clm.jsonl` | **340 行**，回复、耗时、token、评分及部分服务端指标 |
| `data/results-lcm.jsonl` | **340 行**，对照运行的对应字段 |
| `data/workload.jsonl` | 确定性任务序列 |
| `data/boxlog-clm.txt`、`boxlog-lcm.txt` | 公开服务端日志 |
| `docs/recovery-episode-cost-fix.md` | 重复 prefill 的定位与修复设计 |
| `hermes_clm/` | 上下文存储、预算、恢复、检索与工具实现 |
| `docs/context-map.md` | 可恢复来源的索引与引用设计 |

模型为本地 Qwen3.8-27B Q4_K_M，使用双 RX9070、131K 窗口。对照双方各执行一个 340 轮会话、44 个延迟回忆 probe。报告中的 CLM 召回为 41/44，对照为 39/44。平均每轮耗时分别为 54.3 秒和 15.3 秒，CLM 的 prefill 时间约为对照 2.76 倍。仓库明确保留中途修复、超时和单次运行限制。[实验报告](https://github.com/John-MiracleWorker/hermes-clm-showcase/blob/main/benchmarks/reports/2026-10-14-long-horizon-breakdown.md)

本次下载了两份 results 和 workload 并核对行数。results 保存的是每个任务轮次的回复与指标，完整每调用上下文及私有 `events.jsonl` 尚未公开。报告存在“clean rerun”字样，但详细记录和 README 明确说明最终 340 轮仍包含修复前后两个阶段，当前应按混合配置运行解读。早期 workload 也有变更记录，单次双臂结果只能用于发现工程现象。

报告暴露的现象包括恢复阶段重复改写提示造成多次大规模 prefill、token 估计膨胀、消息 envelope 不匹配，以及请求输出上限未正确传递导致长生成。README 另披露两项开发时应用的 Hermes host 修复没有包含在这个公开快照中，直接集成时仍需核验 host 兼容性。[项目 README](https://github.com/John-MiracleWorker/hermes-clm-showcase)

Git clone 本次遇到 HTTP2 传输失败，随后通过 GitHub connector 取得 README 和报告，通过 raw 文件入口取得三份数据。原始文件保存在 `evidence/expanded/hermes-data/`，报告保存在 `evidence/expanded/hermes-report.md`。

## 3. 新增 Skill、作者答复和技术阅读材料

| 入口 | 信息与证据性质 |
|---|---|
| [krema/agent-skills context-management](https://github.com/krema/agent-skills/tree/main/skills/context-management) | 可阅读的社区 Skill，含证据参考与评估说明，讨论恢复原件、上下文选择、编辑成本和继续执行。结合多篇研究编写，CLM 正式实验对其效果没有验证 |
| [SugiV 深度解析](https://blog.sugiv.fyi/context-language-models-dissected) | 与上述 Tiny CLM 配套的逐节论文分析，讨论 FLOPs、stepwise GRPO、效率奖励及限制 |
| [Hugging Face 作者讨论](https://huggingface.co/papers/2609.37725) | 作者解释逐次快照、轨迹中的编辑命令、ATIF-CTX，以及采用解析 FLOPs 的理由 |
| [Hermes 功能提案](https://github.com/NousResearch/hermes-agent/issues/129584) | CLM-style context 模式的需求讨论，issue 状态和文字本身不能证明已上线或测得增益 |
| [DataSci Ocean 解读](https://datasciocean.com/en/paper-intro/context-language-models/) | 方法、实验与适用条件的阅读材料，未定位该文章独立运行的实验产物 |
| [Maximem 解读](https://www.maximem.ai/blog/context-language-models) | 讨论单任务上下文与持久记忆的工程边界，权重发布状态应以当前 HF 文件树为准 |
| [Richland 解读](https://richland.au/blog/metas-context-language-models-agents-rewrite-their-own-context-as-a-file-and-a-stale-kv-cache-patch-makes-the-edits-cheap) | 收集了若干讨论入口，其中 ContextBench/权重未发布的描述已经过时 |
| [课程复现选题](https://github.com/baojian/llm-26-fall/blob/main/docs/project-candidates.html) | 给出合成保留/更新任务比较方案，当前查看的是项目建议，未看到完成实验记录 |

作者在 HF 讨论中确认模型每次调用的上下文由 `context_snapshots/` 保存，编辑命令随 trajectory 记录。社区要求的“每次编辑独立保存 diff 和显式原因”被作者视为可改进方向。该答复帮助定位可审计字段，完整论文运行档案的下载入口仍未在本轮找到。[作者原答复](https://huggingface.co/papers/2609.37725)

技术博客的实验数字需要回到其引用的论文或具体实验仓库核验。本轮主要新增可实际取用的资源为 Tiny CLM 的代码、94 份适配器和逐调用轨迹，以及 Hermes 的逐轮数据与故障分析。原论文最终演化 Skill、全部正式 rollout、raw 训练状态缺口仍保留。

## 4. 本次核验范围

已克隆 Tiny CLM，读取主要报告与配置，解包全部 15 份轨迹文件并核对字段，重算 R1/R2/R3 高压力准确率，将 94 个权重清单哈希与 HF LFS 元数据比对。Hermes 的两份结果与 workload 已下载，结果行数已独立核验。模型训练和推理未在本次重新执行。

新增资源索引见 [EXPANDED_RESOURCES.csv](EXPANDED_RESOURCES.csv)。原始元数据与核算结果见 [AUDIT.json](evidence/expanded/AUDIT.json)。原官方资源继续见 [RESOURCE_CATALOG.md](RESOURCE_CATALOG.md)。
