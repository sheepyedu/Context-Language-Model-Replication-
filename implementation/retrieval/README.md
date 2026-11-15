# BrowseComp-Plus固定语料检索组件

本组件仅完成独立CPU验收，尚未接入任务，也未开始830题模型评测。当前C2训练仍使用原网页工具。

- 官方BM25索引固定为 `b3f37f70c33829eb09d04784a54277a31871fd63`，15文件共2170624147字节。
- 100195篇索引文档逐ID和全文与服务器已冻结官方Parquet完全一致。
- Pyserini1.2.0所带Anserini1.1.1 SimpleSearcher原生BM25默认值，使用官方Java类。
- 每次最多10结果，片段最多512个Qwen Token，文档阅读最多8192 Token。Unicode前缀12检查通过。
- Java Temurin21.0.13+9与Python环境均在项目私有env，空CUDA，不修改训练依赖。

`fixed_corpus.py`提供 `FixedCorpus(index_path, tokenizer_path)` 的 `search(query)` 和 `fetch(bcp://docid)`。

`prepare_retrieval.py`是本服务器路径的资产逐SHA校验及Java解包入口，输入清单在 `locks/RETRIEVAL_LOCK.json`。首次准备会拒绝覆盖已有Java目录，恢复时应先复用实际安装与验收记录。`retrieval-runtime.sh`记录实际CPU运行环境，部署其他路径时修改私有目录。

`check_fixed_corpus.py`用通用查询Paris France检查接线，再对100195篇官方源逐项比对，未读取测试题或用模型生成。小证据见 `evidence/FIXED_CORPUS_ROOT_ACCEPTED.json`。本组件与未来gateway/任务路由的集成仍待完成，T09保持未完成。

本方选择官方BM25作为原文未明确的检索器复现默认值，后续全部对照沿用同一个固定组件。论文所述top10/512/8192的输出上限已落实，不能据此声称检索排序与作者未公开的服务完全相同。

[官方索引](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes/tree/b3f37f70c33829eb09d04784a54277a31871fd63/bm25) · [官方Benchmark仓库](https://github.com/texttron/BrowseComp-Plus)
