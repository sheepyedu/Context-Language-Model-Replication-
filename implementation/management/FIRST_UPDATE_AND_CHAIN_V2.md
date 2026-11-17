# 首个真实更新与后续相邻权重核验

## 当前实测

C2 seed1234在2026年11月17日07:53:45完成首个原生optimizer更新。实际梯度范数0.2782789233，PG loss为-0.0732445664，学习率1e-6，PPO KL为0，TIS为0.999984。08:14完成参数证据补核，08:21建立policy1的native CPU权重锚。当前训练进度为1/70。

实际服务版本为2。原生比较日志中1816条变化记录覆盖227个不同语言权重名称，包括词表输出头和语言层权重，观测最大绝对差为1.9073486328125e-6。记录中存在TP分片与日志重复，这两个计数分别保留。

## 已确认的接收器问题

SGLang的`/weights_checker`在权重相等时返回HTTP200，在权重不等时返回HTTP400，并携带`success=false`及原始张量差异。v4检查器使用urllib的默认成功分支读取正文，实际HTTP400触发异常，原检查器退出1，客户端错误正文缺失。

此外，v4按`language_model.`筛选语言参数。实际Qwen3.5服务使用`model.layers.*.weight`、`model.embed_tokens.weight`和`lm_head.weight`。原生模型的`get_embed_and_head`直接访问`self.model.embed_tokens.weight`与`self.lm_head.weight`，提供了具体的语言组件归属证据。

`native_weight_evidence.py`按实际组件命名识别语言权重，继续要求有限且严格大于零的最大差和平均差。视觉参数变化、旋转位置缓冲变化、无关HTTP400、零差与非有限数值均被拒绝。

## 首次比对原证补核

v5读取原控制日志1461103—3691951字节，保存2230848B原始区间。该区间含原INTENT同秒的四个TP比较入口、张量差异异常及原HTTP400完成，SHA为`f9d53e8e56dc3a13575b6d637edc29c45d68a3065c7e21d3ff7590d41646e6b4`。

`first_update_reconcile_v5.py`绑定原INTENT、初始snapshot回执、准确owner PID/start/argv、实际模型源码和前后模型版本2，读取这份原始服务端异常日志完成验收。此步骤新增两次模型信息GET。原比对POST仅执行一次。

首次HTTP400客户端正文仍然缺失。验收证据类型明确登记为`original_native_server_comparison_exception_log`。原v4 INTENT、请求、退出1及正文缺口均保留。完整原证和参数记录留在服务器，公共仓库提供代码、SHA与聚合摘要。

## 后续相邻检查

`update-chain-prepared-v2`接入v5首更新证明，继续复用原v4的有限loss/正梯度解析和准确owner校验。原`validate_prefix`的AST保持一致，包含缺失步骤、重复冲突、owner边界和版本不相邻等拒绝条件。

接收器先保存原始HTTP正文与整数状态，再解释结果。只有compare操作中的有效、有限、正语言权重差异才能接受HTTP400。snapshot与相等compare要求HTTP200和`success=true`。错误和部分响应保留正文、partial及回执。

实际policy1已执行一次snapshot与一次相等compare，两份原生回执均HTTP200。snapshot回执SHA表示确认消息的字节，具体权重状态由操作、原生实现、准确owner及前后版本2共同绑定。后续policy2/raw3将与policy1锚比较，再建立下一锚。

Mac和Node各通过25项CPU检查，包含8项实际命名空间/原日志检查与17项相邻入口/传输检查。后者使用模拟HTTP响应验证两更新流程，实际policy1锚另有原生回执。CPU检查新增模型、Judge和GPU调用均为0。

## 后续执行

policy1已有实际INTENT与接受记录。policy2在真实第二次更新同步为raw3后执行以下原生入口，参数按当前真实owner核对。

```sh
/mount-data/CLM-Replication/env/rl/bin/python /mount-data/CLM-Replication/management/update-chain-prepared-v2/accept_update_chain.py --policy 2 --attempt 13 --owner-start 51919204
```

checkpoint完整恢复、mask/奖励/账本的更新级闭合、连续3次有效更新、70步、500题选模、对照和最终评测仍待完成。当前T06整体与训练稳定性均保持未完成，实施待办5/10，Goal继续。
