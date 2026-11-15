"""Preserve closed model tool-timeout outcomes as trainable token sequences."""
import json
from pathlib import Path

from .rewards import validate_masks

ROOT = Path('/mount-data/CLM-Replication')


def training_view(result, root=ROOT):
    """Create an adapter-only view. Native results and reward values stay intact."""
    sessions = []
    dispositions = {}
    for session in result.results:
        trajectory = session.trajectory
        evaluation = trajectory.metadata.get('evaluation', {})
        outcome = evaluation.get('clm')
        if (session.status != 'ERROR' or not isinstance(outcome, dict)
                or evaluation.get('outcome_reward') is None or not trajectory.traces):
            sessions.append(session)
            continue
        archive = root / 'runs/session-artifacts' / session.session_id
        events_path = archive / 'logs/agent/edit_events.json'
        if not events_path.exists():
            sessions.append(session)
            continue
        events = json.loads(events_path.read_text())
        if len(events) != len(trajectory.traces) or events[-1].get('tool_error') != 'TimeoutError':
            sessions.append(session)
            continue
        results = sorted((archive / 'logs/agent/commands').glob('*/RESULT.json'))
        if not results or json.loads(results[-1].read_text()).get('timed_out') is not True:
            sessions.append(session)
            continue
        attempts = list((root / 'runs/provider-attempts' / session.session_id).glob('*/INTENT.json'))
        if len(attempts) != len(trajectory.traces) or any(
            not (p.parent / 'RESPONSE.json').exists() or (p.parent / 'ERROR.json').exists()
            for p in attempts
        ):
            sessions.append(session)
            continue
        for trace in trajectory.traces:
            validate_masks(trace.response_ids, trace.response_logprobs, trace.loss_mask,
                           [0] * len(trace.response_ids))
        native = {'session_status': session.status, 'trajectory_status': trajectory.status,
                  'result_error': session.error, 'trajectory_error': trajectory.error,
                  'training_disposition': 'closed_model_tool_timeout',
                  'reward_source': 'unchanged original finalized evaluation'}
        dispositions[session.session_id] = native
        # The adapter's COMPLETED means a complete training sequence here.
        # Original task status and errors are restored in sample metadata below.
        view_trajectory = trajectory.model_copy(update={'status': 'COMPLETED', 'error': None})
        sessions.append(session.model_copy(update={
            'status': 'COMPLETED', 'error': None, 'trajectory': view_trajectory,
        }))
    return result.model_copy(update={'results': sessions}), dispositions


def restore_native_metadata(samples, dispositions):
    for sample in samples:
        native = dispositions.get(sample.metadata['polar']['session_id'])
        if native:
            sample.metadata['polar'].update(native)
