"""Check basic retrieval and match every indexed document to the fixed corpus."""

import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq

from fixed_corpus import FixedCorpus


root = Path('/mount-data/CLM-Replication')
asset_proof = json.loads((root / 'management/RETRIEVAL_ASSETS_ACCEPTED.json').read_text())
started = time.time()
corpus = FixedCorpus(asset_proof['index_path'], '/mount-data/Models/Qwen3.5-9B')
hits = corpus.search('Paris France')['results']
assert len(hits) == 10 and all(hit['snippet_tokens'] <= 512 for hit in hits)
document = corpus.fetch(hits[0]['url'])
assert document['document_tokens'] <= 8192
for text in ['你好🌏 café résumé ' * 1000, 'hello world ' * 1000, '']:
    for limit in (1, 7, 512, 8192):
        prefix, count = corpus.prefix(text, limit)
        assert text.startswith(prefix) and count <= limit
        assert count == len(corpus.tokenizer.encode(prefix, add_special_tokens=False))
try:
    corpus.fetch('https://example.org')
except ValueError:
    pass
else:
    raise AssertionError('Fixed-corpus reads must not fetch the live web')
ids = set()
content_digest = hashlib.sha256()
for path in sorted(Path('/mount-data/Datasets/BrowseComp-Plus/corpus/data').glob('*.parquet')):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=['docid', 'text']):
        for item in batch.to_pylist():
            docid = str(item['docid'])
            assert docid not in ids
            ids.add(docid)
            actual = corpus.raw_document(docid)
            assert actual == item['text'], f'Indexed contents differ for {docid}'
            content_digest.update(json.dumps([docid, hashlib.sha256(actual.encode()).hexdigest()]).encode() + b'\n')
    print('verified corpus shard', path.name, 'cumulative', len(ids), flush=True)
assert len(ids) == corpus.num_docs == 100195
proof = {'accepted': True, 'indexed_documents': corpus.num_docs, 'exact_source_contents_matched': len(ids),
         'source_identity_digest': content_digest.hexdigest(), 'index_revision': asset_proof['index_revision'],
         'top_k': 10, 'snippet_token_cap': 512, 'reader_token_cap': 8192,
         'unicode_prefix_checks': 12, 'live_web_read_rejected': True,
         'tokenizer_path': '/mount-data/Models/Qwen3.5-9B',
         'retriever': 'pyserini1.2.0 / Anserini1.1.1 SimpleSearcher constructor BM25 defaults',
         'query_for_wiring_only': 'Paris France', 'benchmark_questions_used': 0, 'model_calls': 0,
         'elapsed_seconds': time.time() - started, 'active_training_source_changed': False}
(root / 'management/FIXED_CORPUS_ROOT_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
