# Token 与完整轨迹保存

截至北京时间2026-11-16 14:44，已闭回执12,710份。已知输入147,621,917 Token，输出2,059,400 Token，共149,681,317 Token。

| 角色 | 输入 | 输出 | 合计 |
| --- | ---: | ---: | ---: |
| rollout | 147373810 | 1945739 | 149319549 |
| judge | 221116 | 113656 | 334772 |
| conversion_validation | 22236 | 0 | 22236 |
| merging_numerical_validation | 4720 | 0 | 4720 |
| judge_calibration | 35 | 5 | 40 |

原1份辅助HTTP错误的usage缺失保持null，完整总数因此为null，已知量单列。缓存包含于输入，思考包含于输出，均不重复相加。训练重复读取同一Token的次数另记sample-summary，不增加生成账单。

所有原始模型请求、响应、Token IDs、logprobs、loss masks、工具调用及返回、编辑前后上下文、Judge原始请求响应均全量保留。9组288条完整轨迹通过原件审计，8组满足训练入场条件。全量逐轨迹表为TRAJECTORIES_1444.csv，权威账本在CPFS runs/token-ledger，原件在runs/provider-attempts、runs/session-artifacts与runs/c2-seed1234/audit。14:41剩余空间约6.10TB。

训练仍0/70更新。07:19至13:58的可选工程gate曾导致6小时39分等待，已解除并用5真实进程验证有期限lease机制。14:39首次训练前向又因注意力后端无可用实现退出1，修复中。原始失败与资源开销保留，货币费用尚未提供。

逐请求原证核验已闭合，12,710份源回执与权威账本逐份一致，12,420份正式provider INTENT全部有账，34份早期工程调用从原completions复核。240份Judge临时路径已按session绑定到CPFS归档，并核对请求、响应与用量文件的原manifest SHA。网页原始证据已保存9,333份检索、18,666文件，全SHA通过。
