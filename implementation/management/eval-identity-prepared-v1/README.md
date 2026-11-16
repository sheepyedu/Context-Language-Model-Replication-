# 后续独立实验的评估身份

原模板 `clm-c2-eval-{rollout_id}-{sample.metadata.qid}` 没有包含 seed。原生 CPU 测试已复现，仅修改 `clm_run_id` 时，不同种子得到同一个 task ID。协调器可能复用先前实验的结果。

`prepare_eval_identity.py` 为未启动实验生成新配置，只替换 `polar_task_id_template`，将完整 `clm_run_id` 写入评估身份。题目、标签、模型输入、奖励、模型与训练参数沿用输入配置。输出使用新路径，已有文件保留。

当前运行中的 C2 seed1234 配置及原始任务 ID 保留。后续 C1、额外种子和其他独立条件在启动前使用此生成器，再通过 `evaluation-prepared-v1/compile_manifest.py --avoid-manifests` 对已冻结身份集合排重。每个完整实验仍需使用独立的 `clm_run_id`。

此目录提供配置生成与原生 producer 的 CPU 验证。实际后续实验的配置、PINS、owner 入口、模型环境及训练验收需在其正式启动时完成。
