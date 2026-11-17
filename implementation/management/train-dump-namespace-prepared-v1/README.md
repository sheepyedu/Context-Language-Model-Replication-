# 按rank区分原生训练诊断文件

## 已确认的问题

原Slime的`save_debug_train_data`在每个训练rank调用，路径由`args.save_debug_train_data.format(rollout_id=..., rank=...)`产生。当前launcher路径仅含`train-{rollout_id}.pt`，TP4的四个rank会覆盖同一文件。首更新实际完整文件只记录rank1，原始rollout另行完整保留。

该改造使用原生已有rank参数，仅将诊断文件名改为`train-{rollout_id}-rank-{rank}.pt`。数据、奖励、模型、梯度计算、同步、保存与评估日程均保持原launcher的字节。生成器要求显式输入SHA，保留原文件，并使用独占创建生成新候选。未来运行输入有变化时，必须按新的实际launcher重新生成和登记PINS。

## 已完成的验证

Mac12组纯CPU检查通过，覆盖唯一路径替换、原四rank冲突、新70×4互斥路径、幂等、单双引号/无引号/含空格路径与无效字段/重复参数。

Node通过同12组加4个原生writer检查。测试直接加载现有Slime写入函数，用CPU将rank依次设为0、1、2、3，四个实际文件都保留原rollout_id、rank与marker。该测试使用模拟的distributed rank，未启动分布式训练或模型。

原实际v14 launcher经过生成器CLI返回0，生成文件`bash -n`返回0。重复覆盖候选、覆盖原文件、输入SHA错误三个真实CLI均按预期返回1。原launcher SHA前后完全一致，原科学owner UID/PID/start/完整argv哈希仍一致。

原launcher SHA为`3e289d4b40700f8d191c80fe5d67afc597997032d8e42c33112283722b81d366`，生成候选SHA为`573cdd622e3170bc646e5574e9db1be9c22930ffd90c6aa8fdb8fbeb668fab60`。唯一替换字段为save-debug-train-data路径。新增模型/Judge/GPU操作0。

## 实际应用时点

当前attempt13持续运行，已生成的文件与在途argv保持原状。新候选尚未绑定或派发为科学owner。下一次原计划checkpoint恢复的受控入口，需要对其实际launcher运行生成器、将源和新参数写入该owner的PINS，再按原恢复合同启动。后续对照的独立launcher同样适用。

```sh
python3 prepare_train_dump_namespace.py \
  --input /absolute/path/to/selected-future-launcher.sh \
  --output /absolute/path/to/new-rank-separated-launcher.sh \
  --expected-input-sha256 ACTUAL_INPUT_SHA256
```

按rank保存增加诊断文件占用。首批单份实测约1.668GB，后续尺寸与可用空间需按实际保存观察。原始rollout、请求和工具留存的合同继续保持。

当前正式1/70，checkpoint实际恢复、连续3次稳定、完整70步及对照评估仍待。这份准备记录不增加已完成训练步骤或正式待办数。
