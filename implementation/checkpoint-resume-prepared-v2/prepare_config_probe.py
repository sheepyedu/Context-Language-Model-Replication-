"""Create a clearly isolated parser fixture, never a training checkpoint."""
from pathlib import Path
import json
root=Path('/mount-data/CLM-Replication')
folder=root/'management/checkpoint-resume-candidate-v2'
fixture=folder/'parser-only-fixture'
fixture.mkdir(exist_ok=True)
(fixture/'latest_checkpointed_iteration.txt').write_text('4\n')
(fixture/'PARSER_ONLY_NO_MODEL_OR_OPTIMIZER.json').write_text(json.dumps({'kind':'parser fixture','training_dispatch_allowed':False}))
script=(root/'management/train-v11.sh').read_text()
old='--load "$RUN/checkpoints"'
assert script.count(old)==1
script=script.replace(old,'--load "$ROOT/management/checkpoint-resume-candidate-v2/parser-only-fixture"')
(folder/'config-probe.sh').write_text(script)
print(folder/'config-probe.sh')
