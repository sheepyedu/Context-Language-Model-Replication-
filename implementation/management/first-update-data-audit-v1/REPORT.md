# 首个真实更新的数据与奖励验收

## 实际范围

2026年11月17日08:44，CPU独立审阅原attempt13完成后保存的`rollout-0.pt`与`train-0.pt`。原owner3060432/start51919204/UID0及完整argv哈希前后一致，训练继续运行。15项Node CPU fixture验证行校验器拒绝身份、Token、logprob、mask、奖励和信用错配。实际审阅耗时46.45秒，独立有界CPU子进程原生wait0，owner与worker精确gone。新增模型、工具及Judge调用均0。

全部256条逻辑轨迹、8组各32条、11095个模型调用Sample都与保存的训练数据逐项对应。核对包含完整输入Token、响应长度、sample index、原rollout logprob、loss mask、按原真实context-edit事件确定的role mask、客观任务/格式奖励、组内任务标准化、效率信用和逐Token双通道advantages/returns。所有浮点数组有限，二值mask与响应长度一致，效率mask包含在loss mask内。原始文件SHA记录在`ACCEPTED.json`。

任务奖励与成功代价从原逐轨迹元数据独立重算，未使用后续表现、人工标签或新评审。逐Token信用严格等于任务标准化奖励加0.25乘以本轨迹效率信用与编辑role mask。

## 首批构成

- 256条轨迹中43条任务成功，16.80%。这是参数更新前的采样批表现。
- 8个接受组中5个没有成功轨迹，其组内方差来自0与-0.2格式失败奖励。原DAPO准入按组合奖励非零方差，因而这些组仍合法入选，主要提供格式约束信号。
- 132464479次输入Token occurrence，1705266个输出/可训练Token。训练计算复用生成Token，其occurrence与provider用量分别保存。
- 编辑role Token共177610。实际有非零效率信用的编辑Token为10697，占可训练Token约0.6273%，占编辑role Token约6.0227%。该首批效率通道信号较稀疏。

这些数字不足以衡量训练收益。需要后续真实更新批、独立验证及task-only对照才能判断效率信用的作用。当前训练条件和奖励公式保持原计划。

## 留存限制

当前原生debug路径`train-{rollout_id}.pt`缺少`{rank}`，多个TP rank写入同一路径。实际留存文件可完整读取，记录rank1，本次接受仅覆盖该份保存的全批buffer及全部原始rollout。其余rank的独立buffer没有分别保留。原生每rank的原始梯度数据无法由这份文件独立证明。

原始rollout、请求、响应和工具完整留存。后续有必要的受控checkpoint恢复入口应将debug训练文件名加入rank，避免诊断文件覆盖。当前在途科学source、参数和已执行结果未修改。

首次元数据读取虽然输出了完整JSON，进程在大Python对象清理期间未形成成功终态，SSH255与原输出均保留。正式验收改为独立有界CPU子进程，由管理父进程取得实际wait0，结果和原始文件保持只读。

当前正式更新1/70。该数据验收补齐首更新的实际mask与奖励证据，checkpoint恢复及连续3次稳定验收仍待。

独立root首轮因本机把原紧凑JSON重排为缩进JSON，OWNER_EXIT的字节SHA不一致而拒绝。原拒绝留存。随后直接复制远端原始JSON字节并重新登记v2 PINS，未改原退出、候选或科学数据。
