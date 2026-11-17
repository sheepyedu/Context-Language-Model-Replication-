# checkpoint恢复的数据游标与跨Ray进程边界

## 本轮查明的两个问题

原生DataSource.load遇到游标文件缺席会直接返回，缺少字段会取0或空字典。CPU实际调用原save/load方法，复现缺文件与部分字段均可留下默认游标。恢复需要在新任务准入前核对保存文件与实际内存游标。

旧prepared-v2通过args.start_rollout_id选择第一次恢复。然而RolloutManager先在Ray另一进程创建，接收args.start_rollout_id=None，之后训练主进程从模型加载结果得到5并只修改其本地args。原RolloutManager.update_policy_version只调用自定义hook，没有回写其args。真实CPU Ray actor复现远端仍None，旧prepared边界的恢复调用0次。

新版本在接收者进程保存第一次实际权重同步的逻辑policy。该值由原生update_policy_version hook传入，后续同步仍更新当前policy，同时保留首同步值。新CPU Ray实测同步5、6、7时首同步值保持5，恢复边界调用一次，cursor校验先发生。该actor分配GPU为0、CUDA_VISIBLE_DEVICES为空，实际PID3397427/start56673750已精确gone。所有HTTP入口和模型均为CPU fixture替身。

## 改动

- 在原prepared-v2 bridge中只新增首同步policy状态，调整两处控制函数，并在非零恢复时调用checkpoint_cursor.record_loaded_cursor。
- 游标helper读取原checkpoint的rollout/global_dataset_state_dict_{policy-1}.pt，核实际选择的iteration、当前恢复flags、五个必需字段与实际内存值，以及读取前后原文件一致性。审计按policy和PID记录，重复相同证据可复用。
- Ray中序列化的start_rollout_id可以为None，显式给定的值需要与实际同步policy一致。来源为原生权重同步hook，具体任务、qid、答案和成绩不参与此过程。
- 其他bridge函数AST逐项相同，可反向撤销这三项控制改动得到原prepared-v2全文。模型提示、工具、奖励、采样预算与准入题序保持原定义。

当前科学source-training-v10和原owner继续运行。本目录是未来checkpoint恢复的准备，正式source包、PINS、rank分文件launcher与owned恢复入口仍需在实际保存点组合绑定。

## 验证范围

Mac和Node各24组cursor纯CPU检查，以及各37组实际bridge边界AST检查通过。额外4组Node原生save/load CPU覆盖完整往返、epoch重排、缺失文件静默默认值与部分字段静默默认值。另有一个真实Ray CPU actor跨进程参数与hook接线测试，恢复调用旧0/新1，首policy5保留至当前policy7，实际actor退出且无GPU资源。

这些测试覆盖控制与游标I/O。真实Qwen模型、分布式optimizer、训练RNG和完整checkpoint恢复仍待首个原生保存点。

## 原生时序及恢复执行

原生顺序如下。

```text
本批训练 → 保存模型/optimizer/RNG与dataset cursor → 服务权重同步 → 验证 → 下批采样
```

当前save_interval=5且async_save=false，第五次真实更新对应rollout_id=4，下一逻辑policy5，游标文件index4。实际checkpoint迭代、分片文件和保存完成证据须届时读取。

1. 核原生保存成功、数字iteration、全部分片元数据与实际文件、optimizer/scheduler/RNG和cursor4。保存文件、元数据与摘要登记SHA及实际空间。游标再与原batch journal的after状态核对。
2. 新入口的实际parser结果必须选择已保存的训练checkpoint，finetune/no_load_optim/no_load_rng均为false。当前首次初始化这三个字段为true，来自原生初始模型分支，完整恢复入口需核其实际变化。
3. 在任务准入边界使用已有有期限FD租约。仅在原模型请求/工具/任务在途全部闭合并确认暂停回执后，收口精确原owner、child及其自有Ray角色。共享的保护coordinator与GPU0–3保持各自管理。原已消费任务、预算和未知请求全部保留，已闭结果复用。
4. 合并实际新source/PINS、首同步policy修复、游标审计和rank文件名修复。启动新owned恢复入口前核原控制终态及资源，断连先核实际效果。
5. 原生loader实际恢复各rank模型、optimizer/scheduler及训练RNG，记录实际checkpoint iteration和全rank返回的下一policy，检查恢复参数。首个generate的游标审计通过后再恢复网关准入。原batch journal已启动结果按原身份复用。
6. 新服务的原生权重版本计数由新进程开始，需建立与checkpoint逻辑policy的独立映射。旧owner的raw版本序号和参数锚继续保存。模型、optimizer、训练RNG及cursor的实际加载证据完整后才验收T06。

当前SGLang服务的生成RNG没有持久化为训练checkpoint的一部分，重启后的完整生成序列不具备逐位相同的证明。Megatron训练RNG和数据游标分别按原生保存恢复核验。

原始release模型的common元数据含iteration1、无optimizer和RNG，这是初始转换资产的元数据。正式更新计数继续由原生optimizer指标及实际权重同步证据判定。

当前1/70，T06整体和连续3次稳定未完成。该准备没有新增科学任务或增加完成计数。
