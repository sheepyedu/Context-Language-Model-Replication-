> 2026-11-17封存更新。用户已提前结束本次复现，实际验收1/70次更新，第二批提交前取消，专属作业停止。完整冻结框架、证据与缺口见 [封存包](../archive/2026-11-17/README.md)。以下内容保留其历史版本与观测时间。

# 2026-11-16 20:37 当前状态

两项可选优化均未通过各自固定梯度门，已撤回。原v11配置/training-v10科学包继续原256轨迹/11095样本，实际attempt13于20:35:58单次启动，当前模型初始化。原最大微批反向证明复用，无新普通暂停gate。正式更新0/70、5/10，Goal active。完整发现与原数值见management/STRICT_MERGE_NATIVE_REJECTION.md。下方为历史时点记录。

# 当前状态更新

2026-11-16 20:21 严格前缀合并真实TP4首片段未通过逐参数梯度阈值，已拒绝启用。原256轨迹/11095调用样本保留，正式attempt12单次启动，继续未合并数据。只改变activation checkpoint重计算分组1→4，须在原最大微批通过紧梯度与显存验收后进入原参考/PPO训练。正式optimizer仍0/70、5/10，Goal active。详见management/STRICT_MERGE_NATIVE_REJECTION.md。下方均为各自时点的历史记录。

# 实施状态

最新完整观测为北京时间2026-11-16 16:47。待办5/10，真实优化器更新0/70。当前attempt11，owner2874641/start50427236，四rank真实34816-token微批前向与反向均通过。峰值allocated每rank68064294400B、reserved77332480000B，有限正梯度、参数未变、RNG恢复及清理通过。该诊断optimizer0。正式参考前向267/4194，热态约1.99s/微批，仅此段预计剩约2h10。完整optimizer、参数变化和checkpoint恢复仍待验收。

用户最新授权物理GPU4、5、6、7。原两卡不足后使用TP4/DP1同置分时，Qwen3.5-9B、8题×32轨迹逻辑batch、每批1更新、70更新目标及科学超参数保持。GPU0至3由其他任务占用。当前复用training-v10科学源码，训练worker单独启用TileLang依赖，SGLang原依赖保留。

两卡attempt09在最大保留微批反向诊断OOM。四卡attempt10通过34816-token微批前向，反向被FLA针对Hopper与Triton>=3.4的已知错误梯度保护主动拒绝。已按支持的TileLang后端修复，合成GDN前后向数值检查通过后，于16:27:18单次恢复attempt11。原轨迹、失败、预算和费用均保留，新生成及工具重放0。源闭包4107文件PINS为3c01867204986cc3a276be33f5d3deebadbfffbeb6e11bbe87062ea2dafee0dd。

下面均为带原始观测时刻的历史记录，详细本轮修复见management/FOUR_CARD_RECOVERY.md。

## 已闭验收

- 初始转换权重，两个rank各332个语言张量/缓冲与官方HF重载完全相同。4份相同前向的1352位置logprob差0。视觉项及不同温度前向负结果保留，未泛化成不同推理/训练kernel已等价。
- 10项奖励、7项真实Slime样本/loss CPU检查通过。首个32轨迹组937调用、20操作编辑回合的原输入/Token/logprob/mask/成本和全归档SHA通过。
- 原组4份评审因thinking占满4096输出且判词空，原成绩null/费用保留，整组排除，未重评。后续显式关闭judge thinking，2新公开正反例校准及首正式新评审返回完整。
- 原一条已闭工具超时有完整6请求/664生成Token及原负奖励，却被适配器因native ERROR清零。新派生训练视图保原ERROR/负奖励，4回归通过。API未知和缺评审仍invalid。
- 首次网关交接漏等postrun归档完成，原临时目录233文件已全恢复且原SHA一致，中断副本保留。4回归补足退出前所有队列/inflight清空门。

## 偏离与局限

v3采用官方操作编辑判据，即镜像变更、退出0、无输出。它无法证明任意Bash无静默副作用，v2计划的额外语义分类未实施。所有对照保持同规则，未用受限命令菜单。训练用既有公开网页broker，检索质量与论文工具差异须在效果报告披露。评审thinking关闭仅影响后续新任务，原截断保留。

前3次启动失败分别为会话输入类型、CUDA预加载、router生命周期，正式生成0。attempt04原生exit134为修正适配器时完成gateway drain后的作用域退役，已生成数据与预算精确续接，optimizer0。attempt05后在原policy0作完整gateway drain后退役，原exit134保留。attempt06于04:07单次接续，04:21 owner/router/训练子进程实际live，gateway已resume。原model/tool replay0。

## Token与轨迹留存

02:50快照2469次已闭调用，输入27093549、输出427184，共27520733 Token。Rollout27354361、DeepSeek评审144096、转换校验22236、独立评审格式校准40。工程、失败、丢弃、验证与对照均纳入唯一callID账本，生成在途另列，缺usage与货币null。cache/thinking为输入输出子集，训练重复读取Token另表。

CPFS当次约5.95TB可用，保留全部请求/响应、Token/logprob、工具输入输出、镜像前后与评审。第一完整32组已核。早期工程成功样例部分临时辅助文件被cleanup删除，模型轨迹仍在，缺口保留。最终至少3完整组的交付条件继续执行。

完整原件留服务器 `/mount-data/CLM-Replication/runs`。小证据位于 `evidence/`。当前GitHub状态只陈述已核范围。


新增首有效组group004，同题32条完整轨迹，1365完整Qwen调用，32有效评审，6成功/26负结果。1条原native ERROR工具超时及原负奖励保留。全部归档SHA、Token/logprob/mask、21操作编辑回合及成本重算通过，整组满足奖励方差条件。此组验收新增模型0，仍需凑齐8有效组再执行首个参数更新。


03:02新增逐轨迹CSV及角色/任务/策略分账，各维度与总账一致，2881已闭调用32299581 Token，含DeepSeek151472。原始网页检索2025份/4050文件也已复制服务器逐SHA核验。生成与评审在途仍持续，快照随读取更新。统计脚本 `management/token_report.py`，网页辅助记录镜像脚本 `snapshot_web_evidence.py` 与 `receive_web_evidence.py`。


03:30原owner/router/训练子进程live，3个原生32轨迹组已闭，96轨迹/2943完整Qwen调用已做归档与样本审计。其中2组完整有效，首组4评审null保留。两个有效组的61次实际API评审均显式关闭thinking，另3条空答案负结果没有发起API，已逐项区分。参数更新仍0/70。3906已闭调用44087214 Token，DeepSeek175343，缺总usage0，货币null。固定语料检索CPU组件100195篇全文校验通过，见retrieval目录，尚未集成或运行测试题。


## 04:21 当前采样与默认值修订

实际参数更新仍0/70。原生3组闭合、6组正在执行或排队，待执行会话156。初始policy0权重snapshot成功，首次更新后将核参数变化与同步，原service weight_version重启从1计，不能直接当全局训练步。

v4按序惰性入站在原冻结16候选中执行需要的前缀，取得8有效组后跳过尚未启动后缀，所有已有intent必须终态。原cursor按完整BATCH.after推进，8×32与70步目标保持。196608模式穷尽检查、9原payload精确对比、7训练回归通过。选择ID的条件等价不构成随机轨迹逐Token相同，首个实际ADMISSION_RESULT和更新仍待验收。

实际训练seed、题单shuffle和SGLang服务全局seed均1234。逐请求没有派生seed，旧方案的rollout42及step/group/sample派生描述已按实际effective args纠正。重启会重置全局生成随机状态，不宣称跨恢复确定性相同。当前采样器参数未因文档纠正而更改。

5710已闭回执，已知输入66411031、输出939090，共67350121 Token。Rollout已知67135255、DeepSeek评审192590、转换22236、独立格式校准40。1次模型Bash将搜索误发到/v1/clm/web，被模型端点缺messages校验拒绝，原usage仍null。已绑定原命令、请求、错误、时间及80条完整模型trace，单列辅助HTTP拒绝。v2管理报告保留原总账null和各维度总和，未改原错误或奖励，5检查通过。

至少3组96轨迹完整审计要求已满足，继续保留全部。当前剩余5.925TB。原始数据及临时会话凭据保留私有服务器，Git只发布源码与必要小证明。


## 06:04 当前采样与队列恢复

attempt07于05:47单次启动，06:04 owner2594495、router2594496与训练子2594695实际存活，原group009完成18/32，gateway推理3在途，model weight_version仍1，实际优化器0/70。旧attempt06在六队列/registry/推理归档完全闭合后退役原exit134保留。

已确认原调度器将等待容量时间计入执行期限，91条会话在provider INTENT/响应/归档均零时TIMEOUT，原结果null不重跑。第一版partial snapshot导出被API缺pending会话UUID阻断，原协调器仍在，未以新UUID重新创建任务。当前training-v7在尚未启动group010前使用原生入站门暂停，既有intent绕过此门，原group009继续原会话对象。终态边界再替换协调器，原计时6项与新终态恢复3项CPU通过，live切换尚待验收。修改未改变10份原任务payload及4个科学函数，7项训练回归通过。严格训练序列合并仍未启用。

06:00逐轨迹CSV包含全部角色及91条零模型TIMEOUT，五维和CSV总和一致。06:04全部8858回执已知102830821 Token，含DeepSeek258002、转换22236、合并数值检查4720及格式校准40。1辅助HTTP原usage null保持，全部金额null。4组128轨迹完整核验，约6.54TB空余，继续保留全部。完整报告见[TOKEN_AND_ROLLOUT_RETENTION.md](TOKEN_AND_ROLLOUT_RETENTION.md)。


## 06:33 排队计时修复已上线

06:30终态边界切换完成，13原任务TaskStatus和会话终态精确恢复，91个无模型TIMEOUT及null原样保留。旧协调器精确gone，原无owned wait，真实退出整数null。新owner2619786/start46842148及child2619793/start46842163实际live，运行已CPU验证的dispatch补丁。原训练和模型未重启，切换生成0，首个新group010的1676输入182输出及完整finish/usage已核。CPU证明只覆盖边界逻辑，尚未用真实任务等待满执行期限验证。

新增原group009全32轨迹/2511调用/151操作编辑通过归档SHA、Token/LP/mask及成本审计。4个原闭合工具超时ERROR保留，成绩23个-.2、9个0，未成功。累计7组224轨迹/9147调用已核，6组通过训练门。真实优化器仍0/70，当前group010运行中。

06:33全9885已闭回执已知115461731 Token，含DeepSeek264992，1辅助HTTP用量null保留。全轨迹继续保留，CPFS约6.534TB。第三次网页原文6748份13496文件已全SHA落盘，原件保留。first_update_acceptance.py仅完成7项CPU解析检查，真实参数变化/同步和checkpoint恢复仍待验收。


## 后续对照条件准备

C1原生参数解析与父环境0.25下的实际Ray CPU覆盖检查通过，训练角色通过train_env_vars显式设置效率系数0。模型、题单、优化器、8×32与70更新保持。真实GPU角色环境将在C1启动后实测，当前未派发C1。源码与CPU证据位于conditions-c1-prepared。

Summary历史压缩组件通过14项逻辑CPU及实际Qwen模板28672→401 Token检查，采用论文附录E的28672触发和100回合。两份固定Codex提示、来源与许可证保留。首次模板检查的BatchEncoding字段计数错误已纠正，原失败保留，生产C2未使用该准备代码。组件采用稳定task-stage ID保护已消费压缩请求，完整Summary执行器/评价器/训练接线仍待，S0/S1未启动，T09未完成。


## 13:58 工程入站暂停超期及恢复

本执行方在07:18为可选严格合并检查创建了普通文件入站gate，缺少独立到期机制。原采样器实际从07:19:34.793等待到13:58:54.044，共23959.252秒，超过原3600秒承诺。13:58精确解除后，原attempt07、原policy0和模型继续group011。可选严格合并未启用。原问题、停顿和恢复记录完整保留。

停顿期间闭合模型回执0。按原两卡服务保留推导约13.31卡时，列为管理性资源占用，货币账单未提供。14:00原owner实核live，推理3在途，GPU4/5各77116MiB、利用41%/56%，CPFS约6.12TB。参数更新仍0/70。

有期限的原生兼容租约已经完成5个真实进程检查，包括SIGKILL和到期。全部测试owner已退出，模型/工具/Judge调用0。使用说明在management/BOUNDED_ADMISSION_LEASE.md，当前生产gate为空。

新增group010的32轨迹、1740模型调用、61操作编辑通过完整归档与Token/LP/mask检查，保留5个原闭合工具超时ERROR，任务成绩27个0和5个-.2。累计8组256轨迹、10887完整Qwen调用已审，其中7组通过训练门，原首组4个评审null仍排除。原91条空TIMEOUT保持。

当前已超过北京时间10点的决策权交接时间，材料性新决定交回用户，原Goal与已授权训练和修复继续。


## 15:18 注意力依赖修复与Token原证覆盖

原attempt07在首次参考前向因强制flash但未安装FA2/FA3而退出1，更新0/70。两个cuDNN替代候选未通过D256反向数值检查，保留失败并未启用。官方FlashAttention2.8.3用当前Torch2.9.1与CUDA12.4编译器在sm90构建，独立target实际前向和Q/K/V全部256维梯度与float32强制MATH SDPA对照通过，最大前向差0.0080104。原attention-backend仍flash，科学参数逐项相同，仅两项debug输出路径更换以保旧件。

15:18:29单次恢复attempt08 owner2815541/start50014296。新training-v8闭包158文件PINS ff3eb1173ea1adf8d2a250b1105aee1dc7459f72e8863b1e846dc0ef31441f70。科学五函数AST相同，管理新增初始权重同步后的只读snapshot。4生命周期与7原训练CPU回归、冻结首批11095原生Sample全字段离线重建通过，选中4至11组、256逻辑轨迹、游标保持，无模型工具重放。启动仍需核实完整初始化和真实参数更新，当前未达到稳定。

原group011新增32轨迹/1145调用/31操作编辑通过完整归档审计，累计9组288轨迹，8训练门可用组。14:44全12710闭回执已知149681317 Token，含DeepSeek334772，1辅助HTTPusage null，金额未提供。12420 provider INTENT全部有账、34早期工程调用从原completions核计、240 Judge临时来源与CPFS归档原SHA核对。网页原文9333份/18666文件全SHA已存CPFS，全轨迹继续保留。

