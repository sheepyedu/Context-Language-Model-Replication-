# 有期限的入站租约

当前 Linux DSW 采样器使用 `Path(gate).exists()` 等待管理性入站暂停。本次普通文件 marker 在聊天续接间隔中保留，造成 23959.252 秒等待。13:58已解除，原训练继续。该记录和派生资源占用保留在 `management/strict-merge-integration-v1/`。

## 机制

`bounded_admission_lease.py` 创建单一租约持有进程，使用操作系统 ALRM 限制最长 3600 秒。它打开记录目录的文件描述符，并原子发布指向 `/proc/PID/fd/FD/唯一租约.json` 的 symlink。

持有进程到期、提前结束或被 SIGKILL 后，描述符关闭，当前采样器的文件存在性判断立即失效。恢复因此独立于后续聊天唤醒。正常退出还会清理自身 symlink。SIGKILL 留下的悬空链接可由下一租约持有者在文件锁内处理。唯一租约文件名绑定本次记录。

该进程仅持有有限期资源，等待操作系统信号。它的工作内容包括发布、到期与释放记录。没有模型请求和任务/GPU轮询。当前生产 gate 已为空。

## 安全测试示例

以下路径是独立示例目录，可在服务器执行。

```bash
python /mount-data/CLM-Replication/management/bounded_admission_lease.py \
  --gate /tmp/clm-lease-example/gate.json \
  --directory /tmp/clm-lease-example/record-01 \
  --duration-seconds 5
```

管理调用方可以用 `subprocess.Popen(..., start_new_session=True)` 持有该进程的实际句柄，并保留 stdout、LAUNCH 与进程身份。进入切换前核对本次发布记录和采样器实际暂停位置。原工程窗口的时钟和期限应随续接保留。

## 已完成验收

五个真实进程场景通过当前 `training-v7` 原生等待函数验收，分别为到期恢复、SIGKILL 自动失效、悬空链接接续、已有持有者的冲突保护及提前释放、超一小时参数拒绝。所有测试持有进程已精确退出，新增模型、工具或评审请求为 0。

证明为 `management/admission-lease-regression-v1/CPU_ACCEPTED.json`。验收针对当前 Linux PID namespace 和原生文件存在性协议，其他运行环境需要相应兼容性检查。
