# CLM 文件结构与使用指南

对应服务器目录 `/mount-data/context language model`。本文按 SOURCES.json 对应的固定源码检查，编写期间没有启动模型、任务或训练。示例命令需要在准备运行环境和模型端点后执行。

## 1. 整体架构

CLM 将模型即将使用的对话上下文写成一个文件，允许模型通过 Bash/Python 修改它。运行器把修改后的内容送入下一次模型调用。Skill 是指导模型如何管理上下文的文字，例如哪些数据应保存到文件、何时删除旧记录，以及怎样找回准确值。

```text
任务输入 + Skill 文字
       ↓
执行模型 → Bash 命令 → 任务环境
                  └→ 编辑 LIVE_CTX_MAIN.txt
                              ↓
                    修改后的下一轮模型输入
                              ↓
                 任务结果 + 逐次调用/编辑轨迹
                    ├→ ICL → 优化 SKILL.md
                    └→ RL  → 更新 Qwen 权重
```

ICL 固定模型权重，优化结果是 Skill 文本。RL 用在线采样的任务奖励与效率信号更新模型参数。官方提供了两条学习途径，完整的“ICL 最佳 Skill + 旧轨迹 → 离线训练 Qwen”流水线还需要另外设计与验证。

## 2. 顶层目录

```text
/mount-data/context language model/
├── README.md                       资源包入口
├── SOURCES.json                    官方来源与固定提交
├── MANIFEST.json                   原始内容文件大小与 SHA256
├── VERIFIED.json                   服务器校验和协作账号读取回执
├── USAGE_GUIDE_ZH.md                本指南
├── FILE_INDEX_ZH.md                 逐文件源码导航
├── paper/
│   ├── Context-Language-Models_2609.37725v1.pdf
│   └── source/                     TeX、表格、图形和参考文献
├── datasets/ContextBench/
│   ├── README.md                   数据字段说明
│   ├── kv_store.parquet            64 个键值实例
│   ├── log_triage.parquet          56 个日志实例
│   ├── needle_retention.parquet    80 个信息保留实例
│   └── sudoku_sketchpad.parquet     56 个棋盘状态实例
└── repositories/
    ├── context-language-models/     官方 CLM 运行与学习实现
    └── ContextBench/               任务、方法 Skill、生成与评分
```

仓库是固定提交的完整受版本控制文件导出，版本见 SOURCES.json。原始 MANIFEST 对应首次上传的内容，新增中文文档另有校验记录。Parquet 与任务仓库保存同一批任务的不同格式。

## 3. CLM 主仓库各部分的用途

```text
context-language-models/
├── README.md、LICENSE、NOTICE        总说明与许可
├── pyproject.toml                   Python 依赖与 CLI 入口
├── assets/                          展示图片
├── clm/
│   ├── examples/run_harbor.sh        通用 Harbor 单题启动脚本
│   ├── tests/                       上下文文本解析测试
│   ├── clm_harness/
│   │   ├── cli.py                   clm-harbor 命令
│   │   ├── clm_agent/
│   │   │   ├── harness.py           执行主循环
│   │   │   └── prompts.yaml         系统提示
│   │   ├── context_env/             镜像文件、读回与编辑接受门
│   │   ├── context_utils/           文本和消息列表转换
│   │   ├── utils/                   预算、恢复、Skill 和用量
│   │   ├── configs/                 bcp.yaml、edgebench.yaml
│   │   ├── task_templates/          不同任务的输入模板
│   │   ├── agent_trajectory_format/ ATIF-CTX 结构、导出和校验
│   │   └── flops_metrics/           prefix-cache-aware FLOPs
│   ├── clm_contextbench/
│   │   └── clm_stream.py            ContextBench 流式适配器
│   ├── clm_icl/
│   │   ├── evolve.py                Skill 优化循环和选择门
│   │   ├── tasks.py                 任务接口、reward/cost、轨迹摘要
│   │   ├── note.py                  对比轨迹与优化证据
│   │   ├── propose.py               模型调用、候选解析与校验
│   │   └── proposer_prompt.md       Skill 编辑模型提示词
│   └── clm_rl/
│       ├── README.md                RL 配方及数据字段要求
│       └── patches/                 ProRL 与 Slime 两个补丁
└── suffix_cache_reuse/
    ├── README.md、assets/            SCR 说明和图示
    ├── sglang/                      SGLang 0.5.16 实现与测试
    └── vllm/                        vLLM 0.19.1 实现与测试
```

| 核心文件 | 阅读时关注什么 |
|---|---|
| `clm_harness/clm_agent/harness.py` | 模型输入构建、Bash 执行、上下文同步、停止条件、快照输出 |
| `clm_harness/context_env/env.py` | LIVE_CTX_MAIN.txt 的写入和读回 |
| `clm_harness/context_env/edit_gate.py` | 编辑是否符合 fit/shrink 条件及拒绝反馈 |
| `clm_harness/context_utils/context_string.py` | CTX_TURN 标记与消息角色怎样转换 |
| `clm_harness/utils/skills.py` | Skill 如何上传，SKILL.md 如何加入系统提示，工具如何进入 PATH |
| `clm_harness/utils/budget.py` | token 提醒、溢出回滚、重试和最终回合 |
| `clm_harness/utils/resume.py` | checkpoint 与恢复 |
| `clm_harness/utils/tokens.py` | token 计算和快照记录 |
| `clm_harness/agent_trajectory_format/models.py` | segment、input_context、branch 和子轨迹的数据结构 |
| `clm_harness/agent_trajectory_format/validator.py` | ATIF-CTX 轨迹结构校验 |
| `clm_harness/flops_metrics/kv_cache_flops.py` | 上下文编辑导致的 prefix-cache 失效成本 |
| `clm_contextbench/clm_stream.py` | 分批输入、READY_FOR_NEXT_OP、方法 Skill 注入、结束协议 |

SCR 属于推理服务优化。它尝试复用编辑后仍保留的后缀缓存，适配版本见子目录 README。第一次理解 Skill 和轨迹时可以先用普通模型服务，后续再单独比较缓存机制。

## 4. ContextBench 文件怎样看

| 路径 | 用途 |
|---|---|
| `clm_exp/skills/` | 34 份方法说明，覆盖 7 种方法 |
| `clm_exp/skills/clm/` | CLM 的四份任务族 Skill |
| `clm_exp/configs/paper.yaml` | 配套任务生成设置 |
| `tasks/<task-id>/` | 已生成的 256 个 Harbor 任务 |
| `contextbench/tasks/*/generate.py` | 四类题的生成器 |
| `contextbench/tasks/*/scaffold.md` | 任务指令模板 |
| `contextbench/graders/` | 各类任务的确定性评分 |
| `contextbench/harbor/` | 打包、流式交互和评分接线 |
| `contextbench/stream.py` | 输入操作释放与结束协议 |
| `contextbench/tokens.py` | token 与模型快照工具 |
| `contextbench/oracle.py`、`feasibility.py` | 参考处理逻辑和预算可行性检查 |
| `tools/audit_trial.py` | 实际输入流、快照与结果审计 |
| `docs/HARBOR_SMOKE.md` | 工程试跑报告和节选 |
| `docs/DEVIATIONS.md` | 公开移植版与论文内部设置差异 |

一个任务目录中，`instruction.md` 是执行器任务说明，`task.toml` 声明种子、预算、超时、镜像及输入流哈希，`environment/` 提供环境和操作流，`solution/` 提供参考解法，`tests/` 提供 verifier、gold 与评分代码。研究模型实际知道什么时，应以模型 snapshot 为准，参考解法和 gold 留在评测侧。

## 5. 怎样使用已有 Skill

先读 `repositories/ContextBench/clm_exp/skills/clm/kv_store.md`。它要求刚收到 SET-BATCH 时将整批数据保存到不同文件，保留短标记，清理旧命令记录。遇到 GET 时从文件中 grep 准确值，并检查编辑接受回执。四份 CLM Skill 分别针对 kv_store、log_triage、needle_retention 和 sudoku_sketchpad。

其他方法目录的 Skill 需要匹配其运行器。例如 ACM 文件提到 manage_context/query_memory，需要对应工具实现。

| 场景 | 参数 | 注入方式 |
|---|---|---|
| ContextBench 官方方法 | `skill=clm` | ClmStream 按任务族读取 `.md` 并插入任务说明 |
| 通用 Harbor Skill | `skill_dirs=/path/to/skill` | 读取目录里的 SKILL.md，追加到系统提示，并上传工具文件 |

ContextBench 默认 `skill=clm` 会校验四份正文的 SHA16。想修改时复制到单独目录，用该目录绝对路径作为 skill 参数。`skill=none` 用于无方法 Skill 条件。正文在初始化时加载，模型随后决定何时执行其中的步骤。

### 5.1 创建独立运行环境

```bash
export CLM_ROOT='/mount-data/context language model'
export CLM_REPO="$CLM_ROOT/repositories/context-language-models"
export CB_REPO="$CLM_ROOT/repositories/ContextBench"
export CLM_WORK='/mount-data/clm-work'
mkdir -p "$CLM_WORK"

python3.12 -m venv "$CLM_WORK/.venv"
source "$CLM_WORK/.venv/bin/activate"
python -m pip install -e "$CLM_REPO"
python -m pip install -e "$CB_REPO[tokenizer]"
```

源码要求 Python >=3.12,<3.14。若没有 python3.12，可使用实际已安装的 Python 3.13。工作目录选择无空格路径，因为当前 SkillMount 会按空白和逗号拆分字符串形式的 skill_dirs。shell 引号不能消除这种程序内部拆分。

阿里云平台已有 Docker 入口可按以下配置检查。

```bash
export PATH=/mount-data/yangyang-h20-2gpu-train-20261107/env/skill100-20261110/runtime/usr/bin:$PATH
export DOCKER_HOST=unix:///var/docker/proxy/docker-proxy.sock
docker info
```

### 5.2 模型端点

以下以本地 OpenAI-compatible 服务为示例，模型名和地址要替换为实际部署值。EMPTY 只用于不要求认证的本地端点。

```bash
export CLM_API_BASE='http://127.0.0.1:8000/v1'
export CLM_MODEL='openai/qwen36-27b'
export OPENAI_API_KEY='EMPTY'
```

服务需支持原生工具调用。ContextBench 默认上下文预算 32768、单次输出上限 8192，服务窗口至少容纳二者之和。当前服务器已下载的模型和实际端点不能只凭这个示例视为已配置。

### 5.3 运行已下载的一个任务

```bash
harbor trials start \
  -p "$CB_REPO/tasks/kv-store-R1000-seed42" \
  -e docker \
  -a clm_contextbench:ClmStream \
  -m "$CLM_MODEL" \
  --agent-kwarg "api_base=$CLM_API_BASE" \
  --agent-kwarg skill=clm \
  --trials-dir "$CLM_WORK/trials" \
  --trial-name kv-clm-01
```

这道题包含 1000 次 SET、24 次 GET、34 个输入操作。已下载 task.toml 的 agent timeout 为 14400 秒，论文配套 README 指定 21600 秒。若要匹配论文超时，需要另行打包。

```bash
cd "$CB_REPO"
contextbench package --all \
  --config clm_exp/configs/paper.yaml \
  --agent-timeout-sec 21600 \
  --out-dir "$CLM_WORK/cb_tasks"
```

首题先核对 Skill 进入指令、输入流正确释放、编辑被接受和 verifier 有结果。做无 Skill 对照时改为 skill=none，并用新的 trial-name。

### 5.4 通用 Harbor 任务加载自己的 Skill

将正文存到 `$CLM_WORK/skills/my-skill/SKILL.md`。下面任务路径需替换为一个普通 Harbor task。

```bash
export MODEL="$CLM_MODEL"
export API_BASE="$CLM_API_BASE"
export TRIALS_DIR="$CLM_WORK/trials"

bash "$CLM_REPO/clm/examples/run_harbor.sh" \
  /path/to/YOUR_HARBOR_TASK '' \
  --trial-name generic-skill-01 \
  --agent-kwarg "skill_dirs=$CLM_WORK/skills/my-skill"
```

ContextBench 使用上面的 ClmStream 入口，以保留分批输入协议。Skill 文字中的路径和工具需要与环境匹配，宿主运行器必须实际把编辑文件同步到下一轮输入。

## 6. 怎样看轨迹和一次上下文编辑

```text
trials/kv-clm-01/
├── result.json                    任务与 verifier 总结果
├── verifier/                      评分输出与日志
└── agent/
    ├── clm_stream.json            Skill 路径、哈希、指令哈希、实际参数
    ├── interactive_stream.json    操作投递、释放命令、拒绝和终止原因
    ├── trajectory.json            模型生成、Bash 调用、环境观察
    ├── context_snapshots/
    │   └── turn-NNNN.json         逐调用实际消息上下文
    ├── trajectory.ctx.json        分段上下文和编辑分支
    ├── usage.json                 调用、编辑、重试、token 和成本
    └── timing.json                耗时
```

按以下顺序追踪一处编辑。

1. 通过 clm_stream 确认 Skill 和配置。
2. 用 interactive_stream 确认当时模型已收到的操作。
3. 打开编辑前 snapshot，确认所需数据当时可见。
4. 找到 trajectory 中的编辑命令、返回状态和编辑接受回执。
5. 比较下一次调用的 snapshot，检查删除、替换和外存标记。
6. 找后续查询和答案，检查是否找回正确原文，结合 verifier 评分。

在 VS Code 选中两份相邻 snapshot，使用“选择以进行比较”和“与已选项目进行比较”。snapshot 的 kind 用于标识调用类型，源码还会输出最终上下文，部分无 kind 的尾部快照可能是重复最终状态。统计实际模型调用时要结合调用记录。

```bash
python -m clm_harness.agent_trajectory_format.validator \
  "$CLM_WORK/trials/kv-clm-01/agent/trajectory.ctx.json"

python "$CB_REPO/tools/audit_trial.py" \
  "$CLM_WORK/trials/kv-clm-01" --json
```

ATIF-CTX 每个 segment 记录 input_context 和 branch，保留上下文被改写后的真实分段输入。只从最终对话向前拼接会失去被编辑掉的信息。官方文件也未保证保存原始 HTTP header、每个 SSE 分片与逐 token 时间戳，分析 token/s 需要另外的传输记录。

当前可以先看 `ContextBench/docs/HARBOR_SMOKE.md` 的公开试跑节选。论文完整正式 rollout 档案尚未找到，运行首题后会得到自己的输出。

## 7. 它如何逐轮优化 Skill

源码阅读顺序为 `tasks.py → note.py → proposer_prompt.md → propose.py → evolve.py`。

| 阶段 | 输入 | 输出 |
|---|---|---|
| 当前 Skill 执行 | 训练题、当前 SKILL.md、固定 agent | 轨迹、reward、cost |
| 对比证据 | 同任务成功/失败，或成本不同的成功轨迹 | 带 episode 和 step 引用的 note |
| 候选生成 | 当前 Skill、任务描述、note、训练轨迹摘要 | 多个完整 Skill 改写及分析 |
| 校验 | proposer 响应 | frontmatter、正文和引用校验，拒绝记录 |
| dev 评估 | 有效候选、独立开发题 | 每题 reward/cost |
| 选择 | 候选与当前 Skill 的开发成绩 | 新 incumbent、gate、历史、Pareto frontier |

通用代码按任务为单位比较平均 reward 差值 d 与标准误 SE。d>SE 时可接受，差异在一个 SE 内时选择同口径成本更低者，且候选所有开发 episode 都要取得评分。无 dev 参数时使用 train 进行选择，并明确记录 dev_equals_train。

官方 proposer 能看到训练任务 reward。它与我们此前讨论的 score-blind Skill 管理条件有差异，接回自己的框架时应在方法说明中明确。

### 7.1 通用 Harbor 优化命令

准备 train.txt、dev.txt，每行一个任务目录绝对路径，两个集合分离。以下是小规模流程示例，成本暂用 tokens。

```bash
python -m clm_icl.evolve \
  --tasks "$CLM_WORK/train.txt" \
  --dev-tasks "$CLM_WORK/dev.txt" \
  --model "$CLM_MODEL" \
  --api-base "$CLM_API_BASE" \
  --proposer-model "$CLM_MODEL" \
  --proposer-api-base "$CLM_API_BASE" \
  --init-skill "$CLM_WORK/skills/my-skill" \
  --cost-metric tokens \
  --rounds 3 --candidates 2 --reps 2 --workers 1 \
  --out "$CLM_WORK/evolution-01"
```

可以将 proposer 换为更强模型。rounds 是优化轮数，candidates 是每轮候选数，reps 是每题每 Skill 重复次数。它们都会影响实际调用量。

### 7.2 ContextBench ICL 的接入点

当前 HarborTaskSource 调用 run_harbor.sh，该脚本固定通用 ClmAgent。把 ContextBench task 直接传入默认 ICL 命令，会漏掉流式协议。

现有 Python 接口允许 `HarborTaskSource(..., script=自定义启动脚本)`。该脚本保留 TASK_DIR、CONFIG、trial-name、skill_dirs 参数，使用 `clm_contextbench:ClmStream` 运行。设置 skill=none 关闭内置方法文字，使本轮候选只通过 skill_dirs 注入一次。随后以 `Evolution(run_dir, train_source, proposer, dev=dev_source)` 驱动。

这是根据公开接口整理的接入路线，当前包中尚无已经验收的 ContextBench ICL 启动脚本。首个 trial 必须核对流式参数、预算、Skill 注入位置和 cost。复现论文原数值还需要精确 train/dev/test 划分及分阶段候选筛选设置，公开通用代码的默认值不能直接视为完整论文配方。

### 7.3 每次修改的输入输出

```text
evolution-01/
├── args.json                       参数
├── state.json                      当前 Skill、历史、分数与 frontier
├── skills/<id>/SKILL.md             初始及所有候选
├── rounds/r<k>/
│   ├── note.md                     轨迹证据
│   ├── proposer_request.txt        编辑模型输入记录
│   ├── proposer_response.txt       编辑模型原始输出
│   ├── analysis.md                 修改分析
│   └── gate.json                   接受与拒绝理由
├── episodes/<split>/<id>/           每题成绩、成本、轨迹位置
├── trials/<split>/<id>/             原始任务输出
└── best_skill/SKILL.md              最终当前最佳 Skill
```

查看修改时，将上一版 Skill、note、proposer_request、proposer_response、新 Skill、gate 和开发任务轨迹连起来。源码重复运行会复用有成绩的 episode，无成绩的 episode 会被重跑，而且 HarborTaskSource 会删除相同 trial 目录残留。分析失败前应先另存运行目录。

## 8. 再次使用优化后的 Skill

通用任务将 skill_dirs 指向 `evolution-01/best_skill` 即可。最后使用未参与优化和选择的 test 集评估，并记录 Skill 内容哈希。

ContextBench 方法加载器读取 `<family>.md`，演化器输出 `SKILL.md`。可以在已验证的 ClmStream 启动器中继续用 skill_dirs，并关闭内置 Skill。若改为复制成 `<family>.md` 后走 skill 参数，正文注入位置会变化，应记录这一条件并验证实际 snapshot。

## 9. 怎样训练 Qwen

### 9.1 在线参数学习过程

```text
OpenResearcher 问题
  ↓ 当前 Qwen 对每题采样一组 rollout
逐调用原始输入、生成、工具返回、上下文编辑
  ↓
任务评分 + prefix-reuse FLOPs + 编辑 token mask
  ↓
按同题组计算任务优势与效率优势
  ↓
每段生成在其真实原始输入下计算策略损失
  ↓
更新 Qwen → 新策略继续采样 → 验证集选择 checkpoint
```

例如第 3 轮模型基于完整日志生成动作，第 4 轮将日志压缩。训练第 3 轮动作时，需要恢复第 3 轮原输入。最终压缩上下文无法替代历史行为当时的条件。

### 9.2 双通道优势

核心公式为 `A[i,t] = r[i] + 0.25 × a_eff[i] × m[i,t]`。r 是组内归一化任务奖励。m 标记只执行上下文编辑的回合对应的 response token。

对同题组内成功且正常结束的轨迹，计算 `a_eff = clip((成功组平均 FLOPs − 当前 FLOPs) / 成功组平均 FLOPs, -1, 1)`。失败轨迹，以及成功轨迹少于两个的组，效率优势为 0。

例如两个成功 rollout 分别消耗 80、120 单位 FLOPs，平均 100，则效率优势为 +0.2、−0.2。乘权重 0.25 后，对应编辑 token 的优势额外增加 +0.05、−0.05。实际训练还要结合任务优势、KL、loss mask 和归一化。[官方 RL 文档](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_rl)

### 9.3 代码及接线缺口

| 组件 | 作用 |
|---|---|
| ProRL-Agent-Server | 外部 Polar rollout server、gateway、slime_bridge |
| Slime | 外部策略训练框架 |
| `0001-slime_bridge-dual-channel-advantage.patch` | 添加 advantage hook，目标 ProRL 8bc67cc |
| `0002-slime-forward-a_eff-and-role_masks.patch` | 将效率优势和 mask 传到训练器，目标 Slime bf9b1a3 |
| `sample.metadata["a_eff"]` | 需要在 reward post-processing 按组计算，补丁只消费它 |
| `sample.metadata["polar"]["role_mask"]` | rollout 构建时逐 response token 标注 |
| stepwise 输入 | 原输入、token ID、生成 token、logprob、loss mask 需要对齐 |
| launch/config | 模型、采样、judge、数据与分布式资源需完整接齐 |

先针对固定版本检查补丁。PRORL、SLIME 变量应指向事先准备的外部仓库。

```bash
git -C "$PRORL" apply --check \
  "$CLM_REPO/clm/clm_rl/patches/0001-slime_bridge-dual-channel-advantage.patch"
git -C "$SLIME" apply --check \
  "$CLM_REPO/clm/clm_rl/patches/0002-slime-forward-a_eff-and-role_masks.patch"
```

训练参数中启用 `--advantage-estimator grpo` 和 `--custom-advantage-function-path slime_bridge.dual_channel_advantage.custom_advantage_function`，设置 `POLAR_DUAL_CHANNEL_W_EFF=0.25`。这些参数需要加入完整 Slime 启动脚本。

论文描述的配置包括 Qwen3.5-9B、3040 个 OpenResearcher prompts、8 prompts × 32 rollouts、70 步、学习率 1e-6、500 题验证选模。其 policy 使用 16 H200，rollout 使用 48 H200。当前服务器开展缩小实验需另定资源规模。README 的 prompt/response 张量上限与论文逐回合输出上限、上下文预算属于不同层级，复现时需共同核对。

现有静态轨迹可以用于行为分析、数据转换检查或另行设计 SFT。复现这里的 online GRPO 需要由当前策略分组采样，并保持输入、行为策略概率、奖励和 mask 对齐。官方当前包没有包含全部生产逻辑和完整论文训练 launch。

外部代码入口为 [ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) 与 [Slime](https://github.com/THUDM/slime)。

## 10. 阅读与实践顺序

1. 看 kv_store.md、该题 instruction.md 和 task.toml。
2. 准备环境与模型端点，用 ClmStream 完成一个任务。
3. 查看一次编辑前后 snapshot、调用返回和 verifier。
4. 加无 Skill 对照，再扩展到其他任务族。
5. 阅读 ICL 五个核心文件，准备分离的数据集并验证 ContextBench 优化接线。
6. 逐轮检查 Skill 修改输入输出，用最终 Skill 做独立测试。
7. 阅读 RL 补丁和外部框架，先核验一组 rollout 的 token/mask/reward/FLOPs，再进行参数训练。

源码的逐文件导航见 [FILE_INDEX_ZH.md](FILE_INDEX_ZH.md)。官方资料见 [Harness](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_harness)、[ContextBench adapter](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_contextbench)、[ICL](https://github.com/facebookresearch/context-language-models/tree/main/clm/clm_icl)。
