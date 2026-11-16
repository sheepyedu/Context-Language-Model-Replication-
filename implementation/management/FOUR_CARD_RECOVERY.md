# 四卡训练入口与性能修复

## 授权与当前状态

2026-11-16用户允许两卡不足时补充物理GPU6、7，当前使用4、5、6、7。现有其他项目的0至3卡保持其运行归属。16:27:18单次启动attempt11，真实optimizer更新仍0/70。所有测量均未证明端到端加速倍数。

## 原因与已验证修改

1. 原Transformer Engine2.10强制flash，但缺FA2/FA3，attempt07在首参考前向退出。官方FA2.8.3在独立target针对当前Torch2.9.1与H20 sm90编译，D256 GQA THD前向和全部256维Q/K/V梯度对FP32 MATH SDPA通过。CUDA12.4编译器与Torch cu128存在工具链小版本差异，实际kernel数值检查已记录。两个cuDNN候选反向后128维零，均撤回。
2. attempt08首参考微批后OOM。原生logprob chunk512减少词表统计临时张量。TP2输出与梯度差0，TP4最大输出差9.54e-7、梯度差9.31e-10，在预定容差内。原batch256和每批1更新保持。
3. 两卡attempt09最大保留微批的诊断反向仍OOM，退出1。用户新增授权后切换TP4，冻结四卡资源映射、源码与全部有效参数差异。
4. 四卡attempt10最大保留微批原始34672token、padding34816token的前向通过。进入反向时分配59289390080B、reserve68679630848B。FLA依据已知Hopper/Triton>=3.4错误梯度问题主动拒绝，要求TileLang。四rank均未完成反向，没有optimizer。
5. TileLang0.1.15、TVM-FFI0.1.12、Z3 4.15.4.0放独立target，仅Megatron训练worker通过train_env_vars覆盖，SGLang仍使用原TVM-FFI0.1.16。实际H8/D128两组序列129与257，FP32递归参考对照输出relativeL2约0.0033，梯度最差relativeL2 0.003709/maxabs0.004672，预先判据0.02/0.05。原生backend dispatch确认TileLang。首次检查脚本在依赖目标尚复制时读取到不完整namespace，失败原件保留，安装完成后实际检查通过。

最大微批诊断没有optimizer，保留原始Tokens和loss mask，校验有限非零梯度，清空梯度、恢复CPU/CUDA及Megatron RNG并核参数SHA不变。完整原生检查仍在新attempt中继续。合成数值通过不能替代真实参数更新。

## 费用与留存

原始256逻辑轨迹转换成11095逐请求训练Sample，含134169745训练token occurrence，多次前向反向重复读取单列，不计作新生成量。当前闭模型账本12710回执，已知149681317 Token，其中输入147621917、输出2059400，DeepSeek评审334772。1辅助HTTP原usage未知、全部钱未知。全部原始请求、响应、工具及评审轨迹保留，已有9完整审计组288轨迹。

本执行方早先可选入站gate实际超期6小时39分，已披露并用有限进程FD lease修复，原生5进程回归通过。当前没有工程暂停gate。严格前缀合并尚未完成完整原生梯度验收，未用于生产训练。

## 后续验收

四rank真实长序列反向、完整参考/actor计算、optimizer有限梯度、参数变化与推理同步依次核实。随后完成checkpoint5恢复和3连续有效更新。CUDA Graph、请求并发与KV缓存优化须按同负载实测。当前无加速倍数结论。

上游已知问题与支持路径见 [FLA issue640](https://github.com/fla-org/flash-linear-attention/issues/640) 及 [官方安装说明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)。
