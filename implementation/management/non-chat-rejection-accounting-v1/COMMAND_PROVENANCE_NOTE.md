# 7份schema拒绝的调用来源补证

原会话终态归档后，7份非chat请求各自精确落入一条原Bash命令的起止时间。命令中的query/url与原请求逐字一致，原命令均将带/v1的模型Base URL拼接/clm/web，形成/v1/clm/web。原提供的web CLI会先移除/v1。命令捕获HTTP异常后返回0，HTTP schema拒绝和原usage null保留。

原始命令步骤为21、22、23、27、30、32、33。原件、时间区间、请求和错误及各命令INTENT/RESULT的SHA保存在私有COMMAND_PROVENANCE.json，公开摘要只记录覆盖与SHA。

同会话正确调用CLI的步骤1、34、35原stdout均含results和retrieval_id。此前正确调用的步骤3、4、5、6、14返回HTTP500，原CLI stderr保存堆栈，具体上游错误正文尚未从本次所读材料定位。该事实需与后来错误URL的7份schema拒绝分别记录。原模型行为、成绩和成本不改，补证没有发送新模型或网页请求。

后续需要核对真实broker与gateway的失败日志留存，在受控恢复时完善错误归档，并保持模型可见协议和原有预算。当前只形成诊断与原件绑定。
