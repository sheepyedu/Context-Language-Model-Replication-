import hashlib
import json
from pathlib import Path
import random
import re
import pyarrow.parquet as pq

root = Path('/mount-data/CLM-Replication')
output = root / 'data'
output.mkdir(exist_ok=True)
pattern = re.compile(r'associated with this task|/home/|\.xlsx|\.csv')

def normalized(question):
    return ' '.join(question.casefold().split())

splits = {}
excluded = {}
for name in ('rl', 'sft'):
    source = root / 'management' / f'{name}-00000-of-00001.parquet'
    rows = pq.read_table(source).to_pylist()
    excluded[name] = [row for row in rows if pattern.search(row['question'])]
    splits[name] = [row for row in rows if not pattern.search(row['question'])]
train = splits['rl']
assert len(train) == 3040
keys = {normalized(row['question']) for row in train}
assert not keys.intersection(normalized(row['question']) for row in splits['sft'])
validation = random.Random(1234).sample(splits['sft'], 500)
for name, rows in [('train', train), ('validation', validation)]:
    target = output / f'{name}.jsonl'
    payload = ''.join(json.dumps({'prompt': row['question'], 'label': row['answer'], 'metadata': {'qid': row['qid'], 'source': 'wAI-org/OpenResearcher-Dataset', 'source_split': 'rl' if name == 'train' else 'sft'}}, ensure_ascii=False) + '\n' for row in rows)
    if target.exists() and target.read_text() != payload:
        raise RuntimeError(f'Frozen data differs: {target}')
    target.write_text(payload)
summary = {
    'training_count': len(train), 'validation_count': len(validation),
    'original_split_counts': {'rl': 3051, 'sft': 3051},
    'excluded_missing_attachment_qids': {key: [row['qid'] for row in rows] for key, rows in excluded.items()},
    'selection': 'exclude unavailable attachment references; validation random seed1234 from distinct sft split',
    'paper_exact_ids_confirmed': False,
    'normalized_question_overlap': 0,
    'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')},
    'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'management').glob('*-00000-of-00001.parquet')},
}
(output/'MANIFEST.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
