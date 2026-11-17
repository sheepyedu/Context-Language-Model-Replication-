"""Check the sole filename edit and the original native writer on CPU."""

import argparse
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from prepare_train_dump_namespace import prepare


def checks():
    original = 'python train.py --save-debug-train-data "$RUN/training/train-{rollout_id}.pt" \\\n  --save-debug-rollout-data "$RUN/training/rollout-{rollout_id}.pt"\n'
    updated, old, new = prepare(original)
    assert old == '$RUN/training/train-{rollout_id}.pt'
    assert new == '$RUN/training/train-{rollout_id}-rank-{rank}.pt'
    assert updated.replace(new, old, 1) == original
    assert len({old.format(rollout_id=0, rank=rank) for rank in range(4)}) == 1
    assert len({new.format(rollout_id=step, rank=rank)
                for step in range(70) for rank in range(4)}) == 280
    assert prepare(updated)[0] == updated
    for quote in ("'", ''):
        text = '--save-debug-train-data ' + quote + 'train-{rollout_id}.pt' + quote
        assert 'rank-{rank}' in prepare(text)[0]
    spaced = '--save-debug-train-data "folder with spaces/train-{rollout_id}.pt"'
    assert prepare(spaced)[2] == 'folder with spaces/train-{rollout_id}-rank-{rank}.pt'
    invalid = ('--save-debug-train-data train.pt',
               '--save-debug-train-data train-{step}.pt',
               '--save-debug-train-data train-{rollout_id!r}.pt',
               '--save-debug-train-data train-{rollout_id:05d}.pt',
               '--save-debug-rollout-data rollout-{rollout_id}.pt',
               original + '\n' + original)
    for value in invalid:
        try:
            prepare(value)
        except ValueError:
            continue
        raise AssertionError('Invalid template accepted')
    return 6 + len(invalid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    count = checks()
    native_ranks = []
    if args.native_source:
        import torch
        assert not torch.cuda.is_initialized()
        spec = importlib.util.spec_from_file_location('original_train_dump_writer', args.native_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            template = str(Path(directory) / 'train-{rollout_id}-rank-{rank}.pt')
            for rank in range(4):
                with patch('torch.distributed.get_rank', return_value=rank):
                    module.save_debug_train_data(Namespace(save_debug_train_data=template),
                                                rollout_id=0, rollout_data={'marker': rank})
            assert len(list(Path(directory).glob('*.pt'))) == 4
            for rank in range(4):
                saved = torch.load(template.format(rollout_id=0, rank=rank),
                                   map_location='cpu', weights_only=False)
                assert saved == {'rollout_id': 0, 'rank': rank, 'rollout_data': {'marker': rank}}
                native_ranks.append(rank)
        count += 4
    result = {'accepted': True, 'CPU_checks': count, 'native_writer_ranks': native_ranks,
              'actual_training_launches': 0, 'model_or_judge_calls': 0}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
