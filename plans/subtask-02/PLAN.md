# 子任务二：CLM 原方法 RL 复现方案

版本为v5，2026-11-16。方法实施与验收同步版，纳入当日用户授权的四卡资源修订。正式rollout已启动，当前参数更新0/70，训练与对照效果尚待验收。v2的原调研内容与来源继续保留，以下默认值修订均明示。

## 0. 先用通俗的话说明要做什么

本子任务训练的是 **Qwen 自己管理执行过程中的上下文**：模型做任务时可以用 Bash 改写自己的上下文镜像文件；我们保存每一次模型真正看到的输入，让任务完成情况和成功条件下的计算成本共同影响模型权重。

它不是我们此前“执行任务后，由另一个模型编辑 Skill 库”的实验，也不是 `clm_icl` 的 Skill 文本进化。论文这两条学习路线分开。子任务二先复现参数 RL，不混入此前 100 题、人工 seed Skill、强制 Skill 加载、题末 Skill 管理器或额外 SFT。

核心流程：

```text
固定策略版本 → 同题采样32条完整轨迹 → 独立评审任务结果
      ↓ 保存每次真实输入/输出/token/logprob/编辑事件
按完整轨迹计算任务优势和成功门控效率优势
      ↓ 只把效率项附加到纯上下文编辑回合的生成token
用原输入训练每个生成片段 → 一次优化器更新 → 同步新权重
```

**结论：现在足以制定并实施方法复现，不需要把作者回复作为前置条件。** 原文明确的配置照原文；官方代码中已有的默认行为固定版本继承；仍未说明的细节采用下面明示的复现默认值。默认值不能填补不存在的 token、logprob、模型权重或评审结果。

## 1. 证据与决策顺序

来源标识：

- **P**：论文正文4.2、附录C/E/F明确描述。
- **O**：CLM 官方 README、补丁、harness 源码。
- **F**：固定版本 ProRL/Slime 的实现或默认参数。
- **C**：社区实际代码/运行记录，不能当成官方实现。
- **D**：本方案主动选定的默认解释，尚未证明等同作者运行。

遇到冲突时：论文对应 RL 实验的具体说明优先于泛化框架示例；官方 RL README 补充论文没有展开的细节；剩余取固定代码默认值。需要改变公开代码才能使两者接通的地方记录 patch，不伪装成原实现。每次正式运行保存最终展开后的配置，不能依赖随版本漂移的隐式默认值。

本轮回查纠正上一轮调查的表述：采样温度、top-p、thinking、单次输出、回合上限、70步、judge和选模办法均已在论文附录E说明，不应继续列为“等待作者提供”。上一轮指出的字段生产、版本兼容和mask传递缺口仍然成立。

### 1.1 原先五类待补信息的处置

| 信息 | 已由公开材料回答 | 本方案如何补齐 |
|---|---|---|
| 完整启动与依赖 | P/O给算法和大量超参，F给通用启动器 | 固定源码提交；构建可工作的依赖锁和镜像摘要；兼容patch自行移植。精确历史镜像未知不阻塞 |
| agent/builder/evaluator | P给CLM预算、回合、judge、rubric；O给Bash原生harness | 用官方CLM harness接Polar；逐请求builder为首选；同一个沙箱负责执行与最终状态评审 |
| a_eff/FLOPs/role_mask | P/O给公式、成功门控、纯编辑回合、效率权重；P给RL前一轮精确token匹配 | 自行写字段生产和适配模块，保留可重算原证；不等待额外算法 |
| 完整训练样例 | C Tiny CLM提供按调用保存及训练样例构造代码 | 先制造合成数值样例，再保存本方真实rollout group作为端到端验收；不把社区样例冒官方样例 |
| loss/归一化/DAPO/失败细节 | O给token-mean、动态采样；F给标准差/TIS；P给失败惩罚的类别但与O措辞有差异 | 固定本文的计权、过滤和失败表；不确定项标D，并保留必要敏感性实验 |

## 2. 检索到的社区复现：采用什么，不采用什么

本次进行了英文/中文CLM复现、GRPO、ProRL及GitHub检索，重新查看原文、官方在线RL README、Tiny CLM作者博客与本地固定源码，以及Hermes公开工程记录。没有定位到可核验的“原Qwen3.5-9B + 原任务 + 原完整recipe”的独立完整复现；这是本轮检索结论，不证明网上绝不存在。

### 2.1 Tiny CLM

来源：Sugi Venugeethan，`sugix/tiny-clm`；本地固定提交 `52a34a8120a7728cfaec523f48c4aa23af6a3f6d`。本轮重新读 `rl/advantages.py`、`rl/segments.py`、`rl/grpo_stepwise.py`、`configs/e3_r2.yaml` 和 `RESULTS.md`。在线raw入口本轮读取失败，代码分析明确基于已保存的固定快照；作者博客在线可读。

可借鉴的做法：

1. 每次调用保存当时的input_ids/output_ids，按原输入训练；严格prefix成立才合并，编辑打断合并。
2. a_eff按成功轨迹计算，再落到编辑mask；保存成功率、成本、保留组数和编辑行为。
3. 固定reference，记录训练种子，保存失败和与原论文的偏离。
4. 同时报告准确率与成本，不能只凭“编辑增多”判成功。

不直接沿用的配置：1.7B、LoRA、SFT warm start、2K窗口、受限OFFLOAD/GREP等动作、4×8采样、40步和更大的学习率。其代码将GREP也纳入管理token，而官方CLM仅给纯镜像编辑回合额外效率优势；其默认loss是先每轨迹token均值再轨迹均值，和本方案全batch token-mean不同。它用总体标准差+1e-8，旧ProRL用样本标准差+1e-6。必须逐项区分。

重要负结果：仓库三种子结果中效率奖励组没有稳定优于仅任务奖励组；作者指出训练种子差异大、合成任务最安全策略同时最省成本。由此采用“至少主对照、记录组多样性、按训练种子报告”的研究设计，不据此修改原公式追正结果。其过滤零方差组后不补足固定数量的实现也不作为本项目DAPO完整实现。

### 2.2 Hermes CLM showcase

它是长会话工程实验，不是Qwen RL复现。公开记录将大量时间开销定位到恢复过程中反复改变提示前部，导致重复prefill。本方案借鉴：保存prompt hash、首个token差异位置、前缀复用长度和服务端prefill指标，检查成本是否来自harness自身渲染。

不移植它的固定compact/archive工具、强制恢复策略或host摘要进入原方法。其局部运行/修复前后混合结果不能作为论文RL增益的独立证据。

### 2.3 博客的使用边界

解读文用于寻找线索，参数回到论文/代码核对。检索中仍有博客声称ContextBench或权重未公开，也有摘要把参数RL误说成不改变权重；这些不进入方案。SCR是独立服务优化，不能因博客称其“必要”就加入本轮主实验。

## 3. 固定方法与配置

### 3.1 论文及官方配置

| 项目 | 主配置 | 来源/说明 |
|---|---|---|
| 起始模型 | Qwen/Qwen3.5-9B，文本路径；固定权重/tokenizer/template revision | P给模型名；官方模型卡为post-trained模型。具体revision实施时解析、冻结，不虚构作者revision |
| 学习对象 | Qwen语言模型参数；无额外SFT、无LoRA | P描述参数RL，D选择全量语言参数更新；不用已RL的CLM checkpoint作起点 |
| 算法 | stepwise GRPO + success-gated efficiency | P/O |
| 每个有效更新 | 8 prompts × 32完整rollouts = 256轨迹 | P/O；不是256个模型调用或训练片段 |
| 更新次数 | 70次实际optimizer update | P；README“一epoch”不作为同时成立的事实 |
| 优化器 | 固定Slime/Megatron的Adam路径，lr=1e-6，constant，weight_decay=.1，betas=(.9,.98) | P/O；实际衰减实现随底座锁定，不临时改成社区LoRA优化器 |
| PPO ratio边界 | [1-.2, 1+.28] = [.8,1.28] | P/O；不是把ratio剪到[.2,.28] |
| KL | loss项系数.01，low_var_kl；reference为初始策略且固定 | P/O；F默认不更新reference，reward-shaping KL另设0，避免重复惩罚 |
| TIS | 启用；clip下限0，上限2 | P启用；上下限F默认 |
| 采样 | SGLang；temperature=.7；top_p=.95；thinking=true | P |
| 每调用生成上限 | 4096，包括实际生成的thinking及动作token | P；不能使用harness的16384默认 |
| 上下文与预留 | 28K；reserve2048 | P；明确数值解释见3.3 |
| 非编辑任务回合 | CLM最多80；Summary最多100 | P；纯编辑免费回合不等于无计算成本 |
| 任务评审 | DeepSeek 4.1f API + DeepSearchQA rubric，二值成绩 | 用户明确变更；论文原评审为GPT-5.4-nano。评审运行在agent外部 |
| 效率权重 | .25，显式环境变量 | P/O；覆盖补丁1.0默认 |
| DAPO候选批 | 每批冻结16 prompt groups，按序取得前8有效组；足够后跳过未启动后缀 | O给16/8，v4的D采用惰性入站；已启动组全部原预算闭合 |
| 选模 | 500验证题最高准确率；一标准误范围内选更早checkpoint | P；不按测试集挑模型 |
| 最终评测 | 830 BrowseComp-Plus；同一judge | P；本方案不扩数据构造，分割/ID由数据准备任务提供 |
| 原硬件 | 16 H200训练 + 48 H200采样 | P；不是本方已具备或已预留资源 |

### 3.2 未完整说明部分的默认决策

| 项目 | 默认值/办法 | 依据和区别 |
|---|---|---|
| 任务优势 | 同题按轨迹去重，(r-mean)/(sample_std+1e-6) | F旧ProRL；不换成新stable留一基线 |
| 额外advantage whitening | 关闭 | F默认；已做组内归一化，且官方custom hook本身直接构造优势。额外whitening会改变.25尺度 |
| loss reduction | 全个逻辑update的有效生成token总和作分母；包含PG、KL | O token-mean的D明确解释；跨microbatch/DP正确累计，不按Trace数均值 |
| builder | per_request；之后仅在数值等价通过时启用严格prefix merging | P逐调用语义 + F现成实现；不以最终改写后的完整聊天重建旧输入 |
| 策略同步 | 采样阶段固定版本，组全部结束后训练，一次更新后同步 | D保守同步；可同版本并发采样，不跨版本混组。异步优化另立配置 |
| 参数精度 | 语言训练/推理BF16；奖励、logprob与关键统计FP32 | D；不用FP8/量化作主复现以减少变量；保留Gated DeltaNet结构 |
| 实际并行 | 用户11月16日新增授权GPU6/7后使用物理GPU4/5/6/7；训练与推理同置分时，TP=4、DP=1、CP=1、PP=1；CPU optimizer与offload | D实际验收布局；原TP2最大微批OOM后扩为TP4，保持256轨迹逻辑batch |
| microbatch | 1个可容纳片段起步，梯度累计维持256条轨迹的逻辑batch | F/D；禁止以256个Trace冒256轨迹 |
| 实际seed | 训练1234、题单shuffle1234、SGLang服务全局1234；逐请求seed缺省 | F/D实际解析值；纠正旧稿42与逐请求派生的未实施描述。服务重启重置全局随机状态，不宣称跨恢复逐Token复现；扩展1235/1236单独分账 |
| top_k、entropy、grad clip | -1、0、1.0 | F默认；不用模型卡另一套采样建议覆盖论文采样 |
| Adam epsilon及依赖细节 | 使用锁定底座解析出的默认值，写入effective_config和锁文件 | D不猜未安装依赖的数值；实施第一步自动导出，非等待用户输入 |
| eval/save节奏 | step0及每5个有效更新保存/验证；最后70也验证 | D；每5步较社区10步更细，但不是作者公开频率 |
| 单标准误规则 | 对最高验证准确率p用sqrt(p(1-p)/500)，最早达到p-SE的checkpoint | P规则的D操作化；并列范围仅用于选模，不代替差异检验 |
| judge生成 | temperature0；固定模板、版本和解析schema；长度上限4096；显式关闭thinking | D；不确定/格式失败不能默认判0。记录API返回的reported_model、请求ID、usage与原始响应；正式采样前用固定公开样例检查rubric解析一致性 |
| 动态采样进度 | 每批冻结16位置、最多16批，按序收集至够8有效组；不足时执行全批 | D有限运行保护；已启动组全闭合，未启动后缀单列，所有完成与失败成本保留 |

### 3.3 需要明确披露的冲突与选择

评审按用户最新决定使用DeepSeek 4.1f API，服务名deepseek、Chat Completions、配置模型ID `deepseek-v4-1-flash-261017`，启动时核实际返回模型。论文原judge为GPT-5.4-nano，此变化可能改变绝对准确率与训练奖励。C0/C1/C2/S0/S1统一使用新judge，比较论文数值时标明差异。凭据通过服务器skill-api入口引用，配置与GitHub只保存服务名及模型ID。

1. **70步与一epoch**：主方案采用论文70次有效更新。8×70=560个有效prompt group，至少17920条进入训练的轨迹；DAPO额外采样量另计。3040训练题不等于每题只出现一次，也不等于70步必定一epoch。D选择固定种子shuffle，按实际候选消耗推进，耗尽可重新shuffle直至70有效更新，记录每个题的曝光次数与epoch计数。候选组按预先确定的ID顺序选入，不能挑最快返回或最便宜的轨迹；批中超过8组的合格组如何使用也记录，默认不跨更新复用旧策略样本。
2. **28K与32K**：P为CLM运行预算，O的32K prompt/32K response是另层限制。D将28K解释为28672，传入官方 `context_budget_tokens=28672`、`reserve_tokens=2048`，代码的strict_target因此为26624。实际单次response4096，服务总窗口32768；超出服务窗口不截断历史送出。正文没有明确K单位及是否已含reserve，因此必须保留此解释；不要声称28672就是作者原CLI。
3. **预算提示/回退**：未给完整RL harness命令，继承固定官方ClmAgent默认：nudge_ratios=.25/.5/.75，persistent_nudge=adaptive，max_num_retry_on_limit=50，command_timeout=120s，observation_max_chars=10000，edit gate=fit，finish=submit；将max_steps覆盖为80、max_tokens覆盖4096。`lm_call_cap=None`由代码按80推导184，但真正循环是否更早终止以实际出口记录。默认行为不是论文已验证的精确RL设置；尤其回退本身改变历史，所有被移除调用仍留审计与成本账。
4. **计数器**：固定harness预算目前是tiktoken估计并默认用usage校准（CLM_TOK_RATIO=1），不是精确Qwen tokenizer。保留该官方默认以减少改动，另记录真实prompt token并实施服务最大长度检查。训练mask和FLOPs只能用真实token，不沿用估计数。若改精确预算计数器，另记实现变更并对所有对照一致应用。
5. **纯编辑回合**采用固定官方harness的可重算操作判据，即上下文镜像实际变更、命令退出0且stdout/stderr均空。该标签不能证明任意Bash没有静默的其他副作用。v2曾计划再加“仅镜像编辑”的语义分类，实施中未加入该分类，因为现有日志不足以可靠证明命令全部语义。v3明确采用官方操作判据并保留全部命令、输出和前后镜像供审计，查询但无镜像变更、非零退出或有输出的回合效率mask为0。任意Bash能力保持，未以受限菜单增加行为边界。此为D默认解释的修订，所有对照沿用相同规则，报告披露操作标签与语义纯编辑之间的局限。
6. **惩罚文字差异**：P写失败工具调用/格式异常；O明确“无工具调用.3，格式失败.2，仅失败轨迹”。采用O更具体定义，每轨迹每类最多一次并相加；不擅自对每次非零Bash退出扣.3。原始成功字段、格式字段、惩罚和最终r分别保存。
7. **零方差组**：依P/O和C已见实现，按任务奖励r判断；全成功且r相同，即使成本有差异也丢弃。记录因此丢失的效率信号。保留全成功组属于后续算法消融，不能偷偷改主复现。

## 4. 训练数学和逐调用数据合同

### 4.1 奖励与优势

对同题同策略版本组g，每条完整轨迹i有：任务成功y_i∈{0,1}、clean_finish、成本c_i及失败标签。基础设施不明、SSE缺尾、缺token或logprob的轨迹记invalid/null，不补0。

默认任务奖励：

```text
r_i = y_i - (1-y_i) * (0.3 * no_tool_i + 0.2 * format_failure_i)
A_out_i = (r_i - mean_g(r)) / (sample_std_g(r) + 1e-6)
```

成功集合S={i:y_i=1且clean_finish}，不足2个时a_eff全0；否则：

```text
mean_cost = mean(c_i for i in S)
a_eff_i = clip((mean_cost-c_i)/mean_cost, -1,1) if i in S else 0
A_i,t = A_out_i + 0.25 * a_eff_i * role_mask_i,t
```

mean_cost必须正且有限；缺失时不把c_i补0。基础设施invalid的完整组不训练；D选择重新采样新的group ID（仍保留旧失败费用），不补发未知请求，避免把不足32的组冒原采样设置。已知有效的模型任务失败必须保留，不按“质量差”淘汰。此组级完整性策略比旧bridge排除invalid后保留剩余轨迹更严格，明确标D。

### 4.2 FLOPs

按P附录C公式，每调用t保存P_t（输入长度）、G_t（所有生成token）、R_t（复用前缀）。主RL口径依附录E：R_t为本调用输入和同轨迹前一调用输入的精确token LCP，首调用为0；不把历史所有请求形成的trie命中或其他轨迹共享cache算进奖励。公式：

```text
U = P-R
F_t = C_token*(U+G) + C_attn*((P²-R²)/2 + G*P + G²/2)
c_i = sum_t(F_t)
```

Qwen3.5-9B按O固定常数 N_body=6.9196e9、C_token=2*N_body、C_attn=4*8*4096，并用实际模型config复算核对；不是直接把9B全部参数乘进去。本文采用“前一输入”而不是“前一输入+生成”的D精确定义，符合附录C关于响应后续再prefill的计算表述；实际服务可以缓存生成，相关实测成本另列，不能混作同一指标。

所有真实发生的调用都计成本：纯编辑、失败工具前的生成、被rollback删除的回合及最终答案。HTTP失败没有生成数据时费用未知，不填零。标准serving为主，不启用SCR；实际GPU时间、wall time、KV命中另列，不以解析FLOPs代替账单。

### 4.3 mask与loss

- 每次模型调用保存原prompt_ids、response_ids、sampler_logprobs、版本、工具实参与结果、镜像前后原文/SHA/diff。
- loss_mask只覆盖真实采样的生成token；输入、工具返回、复制的历史、外加边界文本为0。
- role_mask覆盖合格纯编辑回合的整段真实生成，包括该回合实际生成的thinking、动作、EOS；不只标命令字符串。其他回合为0。
- 以 `(run, policy_version, prompt_group_id, trajectory_id, call_id, token_offset)` 标识，防止片段拆分/合并重计。
- 显式把mask写入 `sample.metadata['polar']['role_mask']`，不能只留在 `polar.trace_metadata`；a_eff写入sample.metadata顶层。
- 任何长度、来源、角色缺失须在入训练前显式报错，不能让补丁自动补零掩盖接线失败。合法无编辑轨迹可全零，必须有“确实无编辑”的事件依据。
- 每个token使用当时真实输入；即使该段后来被删，也保留训练片段，不反向穿过文件编辑操作求导。
- 按有效token数加权汇总所有microbatch/DP分片，PG和KL统一分母。TIS权重用锁定Slime实现：exp(old_train_logp-sampler_logp)裁剪到[0,2]，作为固定权重；PPO ratio使用当前/旧训练策略。禁止缺sampler logprob时默默将TIS置1。
- CP起步1。需要CP>1时，同步切片mask和logprob并用数值测试确认；当前官方mask补丁本身没有保证这一点。

### 4.4 原生运行与失败

模型正常答错、没有调用工具、可判定格式错误和预算终止是可观察的模型结果；API未知、容器丢失、评审失败则没有可靠成绩。必须用原因映射，而不能把所有ERROR/TIMEOUT统一处理成模型0分或一律删掉。预算截止后的工作区若可独立评分，保存其结果并标clean_finish=false，效率优势为0；训练是否纳入由完整token/任务终态决定。SSE未闭的最后生成不补造、其整轨不作为完整训练样本。

## 5. 工程实施：复用和需要补写的模块

固定源码基线：

| 仓库 | commit |
|---|---|
| facebookresearch/context-language-models | 483198229ab7b90c0eed4651306ab3b3505aaf50 |
| NVIDIA-NeMo/ProRL-Agent-Server | 8bc67cc32607c31988406a2de460b350970efc88 |
| THUDM/slime | bf9b1a3bc5dd493000ae8518f737fb606bbe36e7 |

先用这组语义基线完成兼容移植；不直接取当前stable。上一轮已实测CLM两补丁分别适配，但ProRL自带async补丁对该Slime在train_async.py冲突。第一项工作逐hunk移植、记录patch、跑导入/生命周期/同步测试。即使采用同步采样，bridge所需调用接口也须核实，不能只删掉冲突patch宣称修好。

SGLang、Megatron、Transformer Engine、mbridge、fla/CUDA等用兼容底座的默认依赖构建后输出确切锁文件和容器digest；模型revision同样落盘。不虚构尚未安装验证的组合，不通过latest自动漂移。模型转换前后做固定输入logprob比较，尤其检查Gated DeltaNet、thinking模板、EOS和工具参数。

计划新增六个小模块（名称为设计，不表示已实现）：

1. `clm_polar_agent_adapter.py`：复用官方harness，连接Polar会话gateway；任务sandbox由单方拥有，评审读取同一最终工件；原始镜像及被删除历史独立保存。
2. `clm_reward_evaluator.py`：固定DeepSearchQA rubric、judge、原始候选与解析结果；返回y、clean状态和惩罚标签，不把答案反馈agent。
3. `clm_flops_and_edit_events.py`：真实token成本、编辑分类、每调用mask和prompt差异；不使用“编辑次数”作奖励。
4. `clm_group_post_process.py`：去重、成功门控、DAPO筛选/补采、组均值和a_eff；全组原始记录与排除原因保留。
5. `clm_sample_adapter.py`：逐调用样本、mask映射、版本字段、CP切片和必要token-mean reducer；错误显式失败。
6. `clm_run_and_audit.py`：资源布局、冻结配置、checkpoint和权重同步、resume、指标与独立核算；scope只限新实验，不恢复旧项目。

`clm_icl`、ContextBench四个演化Skill、旧Skill-RSI管理器不嵌入本链。任务本身需要的搜索工具说明与访问环境保留，但不加入经测试集调优的“管理窍门”。可复用官方已有prompt，不将其称为作者未公开的原训练prompt逐字副本。

## 6. 服务器与规模调整

这是部署设计，不是本轮资源验收。未来启动前按aliyun-pai-h20 skill检查当前实例、CPFS、GPU/进程、权限、端口与容器配额。本次未SSH、未停任何服务。

以8H20为目标时，候选4卡policy/4卡rollout，训练TP2×DP2、PP1、CP1；推理先TP2，按实际可用资源和显存决定一个或两个replica。启用底座支持的分布式优化器、激活重算、microbatch1。长上下文OOM先降低并发/packing、增加梯度累计或采用受验收的offload；保持8×32、上下文和更新次数。若必须改CP，先完成mask分片验收；若必须减小逻辑batch或改LoRA，另名“缩放方法实验”，不冒主复现。

256个rollout是逻辑batch，不要求同时开256个容器。根据容器配额排队，每条轨迹隔离状态，同一采样批策略版本不动。减少并发应主要影响吞吐，不能改变组数、每题采样数或奖励。

预计耗时不从64H200线性推算。完成一组真实rollout和一次真实更新后，报告prefill/decode时间、有效组比例、GPU利用率、judge请求和单位update耗时，再给区间估算。主条件70×8×32=17920有效轨迹只是下界；DAPO丢弃、评估、基线、故障及多种子另外预算。

## 7. 实验矩阵与选模

优先完成三条训练条件，全部从相同起始权重、题单、judge及采样设置出发：

| 条件 | harness | 训练信号 | 目的 |
|---|---|---|---|
| C0（仅评测） | CLM | 无RL | 测本方起点，不使用官方已训练权重代替 |
| C1 | CLM | task-only，w_eff=0 | 隔离普通RL带来的收益 |
| C2（主复现） | CLM | task + gated efficiency，w_eff=.25 | 复现论文双通道方法 |
| S0（仅评测） | Summary | 无RL | 测summary起点 |
| S1 | Summary | task-only | 复现主要harness对照 |

先在C2通过工程闭环，C1/S1使用同一已验收实现；所有对照仍独立采样，不复制一个条件的成绩给另一个。P附录F还提到Summary+效率奖励曲线，其token归属细节未完整公开；本子任务将它列为可选补充，若运行需另明确“摘要生成token”的mask，不能悄悄冒原完整四曲线复现。

主方法完成后优先做C1/C2的独立训练种子重复（拟1234/1235/1236）；一条种子的正结果只报告单种子。验证集评估每5步一次，最终冻结checkpoint后才做主测试，不凭BCP中间曲线选模。若需要复现论文测试曲线，另列描述性测试，不反馈训练或选模。

论文表格的42.5%和1.34PFLOPs/题是参考值，不是验收必须追到的硬阈值。最终报告实际准确率、成本、配对差异和不确定性；GPU不同、judge版本/默认解释等偏离同步披露。未提升也可以是实现正确的负结果。

## 8. 分阶段任务与可验收条件

以下是子任务二的完整验收条件。实施进度及原始证据以STATE、Notion逐项记录和固定GitHub交付为准。当前T01—T05已验收，参数更新仍0/70，后续训练与对照继续实施。

| 编号 | 待办 | 完成证据 |
|---|---|---|
| T01 | 冻结版本、默认值与冲突选择，生成环境lock | commits、镜像digest、effective_config、模型与模板SHA；所有空值有真实解析值 |
| T02 | 移植兼容patch，安装训练/采样环境 | patch diff、导入与生命周期检查、Qwen转换与logprob一致性记录 |
| T03 | 接入CLM agent及评审 | 同沙箱状态、固定prompt/rubric、真实调用token与工具闭合、一组可核验结果 |
| T04 | 实现奖励/编辑事件/FLOPs与DAPO | 参考样例重算一致，失败/null与全成功组处置符合本方案 |
| T05 | 实现训练样本与loss通路 | mask来源和长度全覆盖、原输入保存、拆段/合并/DP数值检查 |
| T06 | 一次真实更新、权重同步及恢复 | 非零梯度、真实参数变化、服务新policy ID、保存/恢复优化器与cursor一致 |
| T07 | 连续3次有效更新稳定性验收 | 无非有限loss、无未知mask回退、无串策略、奖励与费用可重算；这不是模型效果收敛 |
| T08 | C2 70步与验证选模 | 70实际更新或如实缺口，完整训练日志/候选组与选模表 |
| T09 | C0/C1/S0/S1及必要重复 | 同协议对照、种子分账、不隐藏失败或选择性报告 |
| T10 | 最终评测、轨迹审计与报告 | 独立从原件重算、checkpoint冻结证据、完整性与效果分开结论 |

核心测试至少覆盖：0/1/2个成功；全成功但成本不同；失败惩罚叠加；轨迹拆成不同数量片段；纯编辑/查询/混合动作/拒绝编辑；thinking/EOS；编辑后原输入不变；缺mask/错位mask显式失败；w_eff=0严格退化；DAPO过滤和有限补采；被删除历史仍计成本；global token-mean跨microbatch/DP一致；CP>1若使用则分片一致；断点恢复不重置进度。

不能以loss下降、训练脚本exit0或API200单独验收。每个阶段要有输入→字段→张量→梯度/结果的对应证据。RL不保证增加编辑频率，也不保证效率项一定改善准确率。

## 9. 轨迹、成本与交付物

每个run保存：

```text
manifest/       源码、模型、环境、提示与评审、默认决策和SHA
groups/         题ID、策略、采样序号、成功/排除、reward、a_eff
trajectories/   原始请求/响应、token IDs、logprob、工具、context snapshots/diffs
training/       trace/sample映射、loss_mask/role_mask、优势、loss与梯度统计
checkpoints/    policy/reference身份、优化器、RNG、数据游标、权重同步回执
evaluation/     原judge输出、解析结果、验证选模、最终测试
accounting/     候选/丢弃/失败/评估成本、解析FLOPs、实际时间及已知费用
reports/        配置偏离、方法结论、失败原因、可复算表格
```

统计需分清prompt、group、trajectory、模型调用、Trace、Sample、有效训练token和optimizer step。特别记录有效混合组比例、每组成功数量、实际纯编辑token数量、a_eff分布、重复prefill、mask异常数、未知轨迹比例、版本陈旧度及被DAPO过滤的全成功组数。

工程验收、原任务成绩、效率改善、跨任务泛化分别报告。ContextBench或work场景拓展属于后续子任务，不借本轮结果直接宣称泛化。

## 10. 本轮交付及项目管理边界

本轮交付本方案、`CONFIG_SPEC.json`和`RESEARCH_RECORD.json`。配置是逻辑规格，不是可以直接传入Slime的成品launcher；其中环境/模型锁值须实施时自动解析。此前核查的具体代码风险及补丁回执位于 `reports/clm-rl-audit-20261115/`，本方案对其“信息缺口”的判断作进一步收敛，不改写原测试结果。

用户已确认建立CLM探索根项目及四个直接子项目，子任务一按用户结论完成并提交固定审核，子任务二进入实施，子任务三/四待子任务二效果评估后决定。根项目由杨洋审核。当前方案和配置交付不增加T01—T10的实施完成数。

## 11. 来源索引

- P：论文[正文4.2与附录C/E/F](https://arxiv.org/html/2609.37725v1)。本地 `reports/clm-resources-20261114/evidence/paper-source/paper.tex` 379—411、`appendix_clean.tex` 92—114、332—334、379—385。
- O1：[官方RL README与补丁](https://github.com/facebookresearch/context-language-models/tree/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_rl)，本轮在线main README也已回读。
- O2：[官方harness](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/clm_agent/harness.py)、[预算](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/utils/budget.py)、[FLOPs](https://github.com/facebookresearch/context-language-models/blob/483198229ab7b90c0eed4651306ab3b3505aaf50/clm/clm_harness/flops_metrics/kv_cache_flops.py)。
- F1：[ProRL旧奖励](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/8bc67cc32607c31988406a2de460b350970efc88/src/slime_bridge/reward_post_process.py)、[adapter](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/8bc67cc32607c31988406a2de460b350970efc88/src/slime_bridge/adapter.py)。
- F2：[Slime默认参数](https://github.com/THUDM/slime/blob/bf9b1a3bc5dd493000ae8518f737fb606bbe36e7/slime/utils/arguments.py)、[loss](https://github.com/THUDM/slime/blob/bf9b1a3bc5dd493000ae8518f737fb606bbe36e7/slime/backends/megatron_utils/loss.py)。
- C1：[Tiny CLM仓库](https://gitlab.com/sugix/tiny-clm)、[训练实现](https://gitlab.com/sugix/tiny-clm/-/blob/52a34a8120a7728cfaec523f48c4aa23af6a3f6d/rl/grpo_stepwise.py)、[结果](https://gitlab.com/sugix/tiny-clm/-/blob/52a34a8120a7728cfaec523f48c4aa23af6a3f6d/RESULTS.md)、[作者复现博客](https://blog.sugiv.fyi/tiny-clm-teaching-a-tiny-model-to-manage-its-own-memory)。
- C2：[Hermes工程记录](https://github.com/John-MiracleWorker/hermes-clm-showcase/blob/main/docs/recovery-episode-cost-fix.md)。
- 模型：[Qwen3.5-9B官方卡片](https://huggingface.co/Qwen/Qwen3.5-9B)。


## 10. v3实施修订与当前证据

2026-11-16 02:50，T01/T02/T04/T05达到各自接线与数值验收，T03完整有效组、T06真实参数更新与恢复、T07稳定性及后续效果验收继续进行。首个同题32轨迹组完整归档，937模型调用和20个操作判据编辑回合已重算。4条原评审因thinking占满4096而判词空，保留null并排除整个组，未重评。后续judge显式thinking disabled，2份新公开正反例校准通过、首正式新评审已完整返回。

已闭模型工具超时若保有完整Token/logprob/工具闭合及原负奖励，只派生可训练序列视图，原native ERROR与原负奖励不变。API未知、轨迹不全或评审null保持invalid。真实原例664生成Token在旧适配器被清零的问题已用4个回归修复。全局Token账本覆盖所有候选、丢弃、失败及DeepSeek评审，账本以实际调用去重，模型训练重复读取Token另计。归档交接曾早于postrun完成退出，原临时目录233文件已全部恢复且逐SHA一致，原中断副本保留。

当前实际优化器更新0/70，不能以本节的CPU、完整轨迹或归档验收替代训练完成。GPU仅4/5，70步、8×32、原任务/提示/公式/预算不变。正式运行与全部对照统一使用v3明示的评审和编辑判据。


## 11. v4按序惰性入站

现有两张H20与四个任务执行槽下，原收集器会等完整16候选组结束后才选择其中前8个有效组，未入选的后缀仍发生生成。v4每批仍先冻结16个候选的位置、内容与数据cursor，在原顺序中计算原有资格判据。取得所需有效组之后只跳过没有intent的候选，所有已经发出的组等待其原终态，随后才允许权重同步。

数据cursor仍取原BATCH的after值，选中的8组仍为256条完整轨迹。资格判据、既有UUID、已完成actor/工具/评审和全部费用保留。ADMISSION_RESULT记录planned/executed/unstarted-skipped，跳过项不冒成功、失败或已执行。新增采样的后续所有对照沿用同一种调度。

196608个穷尽选择模式覆盖全部16位有效性组合与三种已有intent集合，验证给定同一组结果时，新收集器选出的前8个有效ID与全批过滤完全一致，并且所有已启动组都会收口。原9个payload逐字段匹配，7项训练通路回归通过。该结论针对选择语义，实际随机生成的逐Token一致性没有被证明，也不作为效果结论。优化器更新与模型效用仍需原T06—T10验收。

v3的全批执行记录作为历史条件保留。切换前等待原gateway全部六项队列/inflight指标为0，精确退役采样等待中的旧控制器，从原policy0检查点与原BATCH/结果/预算续接，新source独立PINS冻结。此阶段没有新增actor或评审探针。


## 11月16日四卡资源与恢复边界补充

用户允许两卡不足时加入物理GPU6、7，实际TP4使用4、5、6、7，其余科学超参数与70更新目标保持。原两卡失败、费用和全部原轨迹保留。官方FA2.8.3与原生logprob chunk512已完成数值/梯度校验，训练worker单独启用TileLang0.1.15支持Hopper上GDN反向。四rank真实34816-token微批的前后向、有限正梯度、参数不变与RNG恢复已经验收。诊断optimizer为0，正式首次完整更新仍待。当前source为training-v10复用与trainer-tilelang-v11配置，PINS 3c01867204986cc3a276be33f5d3deebadbfffbeb6e11bbe87062ea2dafee0dd。

T06中的“第五次更新检查点”按当前Slime源码实际约定处理。训练循环rollout_id从0开始，第五次更新保存的是rollout_id=4。恢复的逻辑policy应为5，数据游标文件是rollout/global_dataset_state_dict_4.pt。不能仅按目录数字5猜测其更新计数。实际checkpoint目录名、iteration、optimizer/scheduler步数与模型更新数需从真实产物核对。

当前初始启动在checkpoint不存在时由原生parser设置no_load_optim/no_load_rng/finetune并回落到初始权重。恢复训练时必须从真实存在的checkpoint tracker及shard启动，确认这些参数不再跳过optimizer和RNG。全局数据游标、模型、optimizer、scheduler与RNG共同闭合后才能验收恢复。服务端全局生成RNG随重建重置的已知差异保持披露。

旧checkpoint-resume准备文件早于queue修复与初始snapshot，不可整份覆盖当前源。新版候选只基于当前冻结bridge做恢复边界和审计文件名修订，其他函数AST逐项一致。当前运行源未改，完整原生恢复仍待。
