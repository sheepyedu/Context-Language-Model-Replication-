"""Namespace future evaluation tasks by the complete experiment identity."""

import argparse
from pathlib import Path

import yaml


TASK_ID_TEMPLATE = 'clm-{args.clm_run_id}-eval-{rollout_id}-{sample.metadata.qid}'


def prepare_config(config):
    run_id = config.get('clm_run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('The configuration must contain its complete clm_run_id')
    return {**config, 'polar_task_id_template': TASK_ID_TEMPLATE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.config.resolve() == args.output.resolve():
        raise ValueError('Write a new configuration for an unstarted experiment')
    config = yaml.safe_load(args.config.read_text())
    prepared = prepare_config(config)
    with args.output.open('x') as handle:
        yaml.safe_dump(prepared, handle, sort_keys=False)


if __name__ == '__main__':
    main()
