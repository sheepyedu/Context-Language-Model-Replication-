# C1 task-only 四卡条件准备

该版本对应当前 C2 的原未合并训练配置，源包为 training-v10，启动命令来源为 train-v14.sh，公共依赖以 C2 的 training-v14 source pins 为准。物理 GPU 为4、5、6、7，TP4、Slime/Megatron、BF16、FA2与训练角色专用TileLang环境均按该基线配置。

与当前 C2 原生 effective args 的差异共9项，分别涉及独立实验身份、输出/检查点/审计/入站路径、配置路径，以及训练角色环境中的 POLAR_DUAL_CHANNEL_W_EFF=0。完整差分在 EFFECTIVE_DIFF.json。8题×32轨迹、70更新、任务与格式奖励、题单、模型及优化器设置沿用基线。

原生参数解析通过。CPU专用Ray工作进程在父job效率系数0.25下取得实际系数0，调用原生优势函数后得到task-only的[1,1,1]，CUDA设备可见性为空、CUDA_initialized=false，进程已精确退出。同期四个实际C2训练角色的/proc环境仍为0.25，来源为training-v10。该检查覆盖环境隔离和优势接线，实际C1 GPU角色验收仍待正式运行。

当前交付为配置与CPU准备，C1科学任务、模型与Judge调用均为0。正式启动需要为C1建立专属owner、CURRENT_OWNER和dispatch记录，核当前C2的真实终态与资源归属，再核实际GPU角色环境和首个完整更新。启动时应以最终采用的C2配置核对匹配关系。

当前C2同一owner继续参考前向，相关进度以STATE和真实进程日志为准。首个实际更新、checkpoint恢复和连续3次稳定仍待完成。
