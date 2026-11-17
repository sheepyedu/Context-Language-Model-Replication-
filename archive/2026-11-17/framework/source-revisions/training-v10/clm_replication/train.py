"""Owned synchronous Slime entrypoint with an exportable effective config."""

import json
import os
from pathlib import Path

import ray

from slime.utils.arguments import parse_args


def main():
    args = parse_args()
    path = Path(os.environ.get('CLM_ATTEMPT_DIR', args.clm_audit_dir))
    path.mkdir(parents=True, exist_ok=True)
    (path / 'effective_args.json').write_text(json.dumps(vars(args), indent=2, default=str))
    if os.environ.get('CLM_VALIDATE_CONFIG') == '1':
        print('CLM_CONFIG_VALIDATED')
        return
    from train import train
    ray.init(address=os.environ['CLM_RAY_ADDRESS'], runtime_env={'env_vars': {
        name: os.environ[name] for name in ('LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_DEVICE_MAX_CONNECTIONS',
                                           'PYTHONPATH', 'HF_HUB_OFFLINE', 'HF_HOME', 'CLM_RESUME_INITIAL_GATEWAY') if name in os.environ
    }})
    train(args)


if __name__ == '__main__':
    main()
