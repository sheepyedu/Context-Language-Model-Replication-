> 2026-11-17封存更新。用户已提前结束本次复现，实际验收1/70次更新，第二批提交前取消，专属作业停止。完整冻结框架、证据与缺口见 [封存包](../archive/2026-11-17/README.md)。以下内容保留其历史版本与观测时间。

# CLM逐调用参数RL实现

本目录是持续实施版本。有效优化器更新与端到端验收进度见[STATUS.md](STATUS.md)。

- `src/clm_replication/agent_runner.py` 使用固定官方CLM harness并记录实际Bash、上下文镜像及编辑事件。
- `polar_agent.py` 提供独立Docker任务执行入口。
- `gateway.py` 记录逐次模型请求与响应，在清理前归档任务文件。
- `evaluator.py` 从原轨迹核对工具与编辑事件，并调独立评审。
- `judge.py` 使用DeepSeek4.1f API与DeepSearchQA rubric。
- `rewards.py` 计算任务奖励、纯编辑效率优势和前缀复用FLOPs。
- `bridge.py` 在固定策略下收集完整候选组、筛选DAPO并保存逐调用训练样本。
- `advantages.py` 严格验证双通道优势字段与mask后接官方补丁。
- `accounting.py` 生成一次性Token回执和按角色、任务汇总。
- `train.py` 提供Slime/Megatron原生训练入口和真实effective args。
- `management/` 保存当前专用服务器入口。依赖本项目CPFS环境与固定上游补丁，部署前需修改路径并配置凭据入口。
- `locks/` 保存实际包锁与源码SHA。模型、CUDA扩展及上游源码需依方案安装。
- `tests/` 保存奖励、训练字段与记账的CPU检查。

测试命令在安装上游依赖后执行。

```bash
PYTHONPATH=implementation/src python -m unittest discover -s implementation/tests
```

服务器需要自行配置 `/mount-data/Tools/skill-api` 的DeepSeek服务。仓库配置仅含服务名和模型ID。当前公共检索由Mac broker反向隧道提供，属于复现默认实现差异，实际网页及工具原返回保留。


当前training-v5新增 `native_failures.py`，只为有完整原证的模型工具超时派生训练视图。原任务状态和负奖励保留。`management/train-v5.sh`/`train-config-v5.yaml` 记录当前服务器入口，部署时将本目录源安装到对应版本路径并复用固定上游源码。不要运行历史launcher来覆盖活跃任务。当前judge显式关闭thinking，全部新对照保持一致。详细变更、失败和未完成项见 `STATUS.md`。

