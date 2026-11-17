# 非chat载荷的原始JSON schema拒绝分账

## 实际发现

2026年11月17日09:05小时检查，usage缺失由原1份增至8份。新增7份属于同一在途会话，其中5份请求action=search/query，2份action=fetch/url。原请求没有messages，原ERROR为json_parse_error/invalid_request_error，消息明确缺少messages。7份原目录均只含INTENT、ERROR、TOKEN_RECEIPT，没有保存成功模型响应。当前会话的原始命令归档尚未提供，本次归类依据为原载荷和原错误，调用方具体命令未作归因。

现有分类器只识别先前独立接受的单一HTTP请求。新通用分类只接受上述明确非chat字段组合、精确错误与原件绑定，不依赖qid、session、call ID或具体query文本。原model/usage/response字段非空、messages存在、报错不同、未知action、目录逃逸、原receipt不一致或额外响应文件均维持原类型。

## 保持的原始事实

原receipt和请求、错误不改。8份HTTP请求的原始usage仍为null，原报告中的known_total=0仅表示没有已知数值，complete_total仍null。原总请求数量和所有已知Token和费用口径保持。所有实际错误、退出、候选、预算与任务成绩原样保留。

报表新增non_chat_payload_schema_rejection类，当前分7份新增拒绝、1份原已接受辅助HTTP、15313份模型及工程验证回执。15313份的input/output/total字段完整。已闭快照累计已知177587670 Token，含DeepSeek验证器325次/408595 Token，货币未提供。该快照逐轨迹、任务、policy、role及请求类型全部可加回总账。当前在途未闭调用等待其原回执。

## 验证与接线

Mac和Node各14组纯CPU检查通过，涵盖search/fetch、类型与schema边界、源目录/原件匹配、错误时间顺序、完整响应存在与真实usage等反例。额外AST比较确认v5仅新增分类导入及该分支，其余v4的聚合、CSV、原unknown和总账逻辑完全相同。

原始8份缺失usage的request/error/receipt均在Node实际读回并绑定SHA，其中包含旧1份和新7份。真实v5 CLI退出0，原多维/CSV总账检查通过。分类组件和报表是离线管理代码，当前训练与源包继续运行。

## 使用

在既有runtime环境内，将原management目录加入Python搜索路径，然后执行新脚本。

```sh
ROOT=/mount-data/CLM-Replication
"$ROOT/management/runtime-gpu4567.sh" env \
  PYTHONPATH="$ROOT/management:$ROOT/src:$ROOT/src/ProRL/src:$ROOT/src/slime:$ROOT/src/Megatron-LM" \
  python "$ROOT/management/non-chat-rejection-accounting-v1/token_report_v5.py"
```

结果生成在新的`runs/token-reports/<timestamp>`，完整逐轨迹CSV和原回执索引留服务器。该分类只解决报表中非chat请求与模型回执的计数区分，具体错误调用的上下文与命令来源仍待该会话终态归档。当前正式1/70，checkpoint恢复、3次稳定及完整实验仍待。
