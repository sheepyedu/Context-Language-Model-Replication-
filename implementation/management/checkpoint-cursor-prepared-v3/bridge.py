"""Strict CLM metadata conversion around the pinned Polar bridge."""

import json
import asyncio
from pathlib import Path

from .rewards import Outcome, group_advantages, validate_masks

_served_policy_version = None
_initial_synced_policy_version = None


def trajectory_outcome(sample):
    polar = sample.metadata["polar"]
    outcome = polar["trajectory_metadata"]["evaluation"]["clm"]
    return Outcome(polar["session_id"], outcome["success"], outcome["clean_finish"],
                   outcome["flops"], outcome["no_tool"], outcome["format_failure"])


def post_process_rewards(args, samples):
    groups = {}
    for sample in samples:
        item = trajectory_outcome(sample)
        groups.setdefault(sample.group_index, {})[item.trajectory_id] = item
    advantages = {}
    for group, items in groups.items():
        result = group_advantages(list(items.values()))
        if not result["accepted"]:
            raise ValueError("DAPO zero-variance group reached the optimizer")
        for session, values in result["trajectories"].items():
            advantages[(group, session)] = values
    raw, normalized = [], []
    for sample in samples:
        polar = sample.metadata["polar"]
        outcome = polar["trajectory_metadata"]["evaluation"]["clm"]
        index = polar["trace_index"]
        mask = [int(outcome["pure_edit_by_call"][index])] * sample.response_length
        validate_masks(sample.tokens[-sample.response_length:], sample.rollout_log_probs,
                       sample.loss_mask, mask)
        values = advantages[(sample.group_index, polar["session_id"])]
        sample.metadata["a_eff"] = values["a_eff"]
        polar["role_mask"] = mask
        raw.append(values["reward"])
        normalized.append(values["a_out"])
    versions = {sample.metadata['polar']['trajectory_metadata']['policy_version'] for sample in samples}
    if len(versions) != 1:
        raise ValueError('Training samples mix policy versions')
    version = versions.pop()
    summary = {
        'policy_version': version, 'status': 'prepared_for_optimizer',
        'trajectory_count': len({sample.metadata['polar']['session_id'] for sample in samples}),
        'prompt_group_count': len({sample.group_index for sample in samples}),
        'model_call_samples': len(samples),
        'input_token_occurrences': sum(len(sample.tokens) - sample.response_length for sample in samples),
        'output_token_occurrences': sum(sample.response_length for sample in samples),
        'trainable_generated_tokens': sum(sum(sample.loss_mask) for sample in samples),
        'efficiency_role_tokens': sum(sum(sample.metadata['polar']['role_mask']) for sample in samples),
        'accounting_rule': 'Training sample occurrences reuse generated tokens and are not added to provider token totals.',
    }
    path = Path(args.clm_audit_dir) / f'sample-summary-policy-{version:05d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != summary:
        raise ValueError('Previously recorded training-token summary differs')
    path.write_text(json.dumps(summary, indent=2))
    return raw, normalized



def candidate_decision(args, candidate):
    group = candidate["samples"]
    unique = {}
    invalid_reason = None
    try:
        if candidate['rejection']:
            raise ValueError(candidate['rejection'])
        for sample in group:
            if sample.status.name in ("FAILED", "ABORTED"):
                raise ValueError("incomplete_session")
            item = trajectory_outcome(sample)
            unique[item.trajectory_id] = item
        if len(unique) != args.n_samples_per_prompt:
            raise ValueError("incomplete_group")
        result = group_advantages(list(unique.values()))
        keep = result["accepted"]
        invalid_reason = result["reason"]
    except (KeyError, TypeError, ValueError) as error:
        keep = False
        invalid_reason = str(error)
    return keep, invalid_reason, len(unique)


def should_admit_candidate(eligible_count, required_count, previously_started):
    """Keep every started task, then admit only the required ordered prefix."""
    return previously_started or eligible_count < required_count


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime_bridge import rollout
    if evaluation:
        if _served_policy_version is None:
            raise ValueError('Evaluation began before the initial weight synchronization')
        previous = args.clm_phase
        args.clm_phase = 'validation'
        try:
            return rollout.generate_rollout_polar_async(args, _served_policy_version, data_source, evaluation=True)
        finally:
            args.clm_phase = previous
    if _served_policy_version != rollout_id:
        raise ValueError('Sampling policy version differs from the synchronized actor')
    import os
    if os.environ.get('CLM_RESUME_INITIAL_GATEWAY') == '1' and rollout_id == _initial_synced_policy_version:
        if rollout_id > 0:
            from .checkpoint_cursor import record_loaded_cursor
            record_loaded_cursor(args, rollout_id, data_source)
        import httpx
        response = httpx.post(args.polar_gateway_url.rstrip('/') + '/admin/inference/resume', timeout=10)
        response.raise_for_status()
        assert response.json()['paused'] is False
        receipt = Path(args.clm_audit_dir) / f'resumed-after-policy-sync-{rollout_id:05d}-{os.getpid()}.json'
        with receipt.open('x') as handle:
            json.dump({'policy_version': _served_policy_version, 'gateway': response.json()}, handle)
    target = args.rollout_batch_size
    accepted = []
    audit = []
    try:
        for candidate_batch in range(16):
            candidates = asyncio.run(collect_candidate_groups(args, rollout_id, candidate_batch, data_source, target - len(accepted)))
            for candidate in candidates:
                group = candidate['samples']
                keep, invalid_reason, trajectory_count = candidate_decision(args, candidate)
                selected = keep and len(accepted) < target
                audit.append({"candidate_batch": candidate_batch,
                              "group_index": candidate['group_index'],
                              "trajectory_count": trajectory_count, "selected": selected,
                              "reason": invalid_reason if not keep else (None if selected else "batch_excess")})
                if selected:
                    accepted.append(group)
            if len(accepted) == target:
                break
    finally:
        args.rollout_batch_size = target
        path = Path(args.clm_audit_dir) / f"groups-{rollout_id:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2))
    if len(accepted) != target:
        raise RuntimeError("Insufficient variable-reward groups within bounded sampling attempt")
    return RolloutFnTrainOutput(samples=accepted, metrics={"clm/candidate_groups": len(audit)})


# The synchronous trainer calls the lifecycle hooks on the entrypoint module.
def update_policy_version(args, version):
    global _served_policy_version, _initial_synced_policy_version
    _served_policy_version = int(version)
    if _initial_synced_policy_version is None:
        _initial_synced_policy_version = _served_policy_version
    from slime_bridge.rollout import update_policy_version as update
    result = update(args, version)
    if version == 0:
        from .weight_provenance import capture_initial_snapshot
        capture_initial_snapshot(args)
    return result


async def wait_for_task_admission(args, rollout_id, candidate_batch, position):
    gate = getattr(args, 'clm_admission_gate_path', None)
    if not gate or not Path(gate).exists():
        return
    directory = Path(args.clm_audit_dir)
    name = f'control-admission-{rollout_id:05d}-{candidate_batch:03d}-{position:03d}'
    paused = directory / (name + '-paused.json')
    if not paused.exists():
        with paused.open('x') as handle:
            json.dump({'gate': gate, 'rollout_id': rollout_id, 'candidate_batch': candidate_batch,
                       'position': position, 'new_task_intent_created': False}, handle)
    while Path(gate).exists():
        await asyncio.sleep(.2)
    with (directory / (name + '-resumed.json')).open('x') as handle:
        json.dump({'gate': gate, 'position': position, 'admission_resumed': True}, handle)


async def collect_candidate_groups(args, rollout_id, candidate_batch, data_source, required_groups):
    """Reuse Polar transport and conversion without speculative next-policy work."""
    import httpx
    from slime_bridge import rollout
    from slime_bridge.config import resolve_polar_slime_config
    from polar.rollout.models import TaskResult, TaskStatus
    from slime.utils.types import Sample

    config = resolve_polar_slime_config(args)
    semaphore = asyncio.Semaphore(int(args.clm_candidate_group_concurrency))
    directory = Path(args.clm_audit_dir) / f"rollout-{rollout_id:05d}" / f"candidate-{candidate_batch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    cursor_fields = ('epoch_id', 'sample_group_index', 'sample_index', 'sample_offset', 'metadata')
    cursor = {key: getattr(data_source, key) for key in cursor_fields}
    batch_path = directory / 'BATCH.json'
    if batch_path.exists():
        batch = json.loads(batch_path.read_text())
        if cursor not in (batch['before'], batch['after']):
            raise ValueError('Dataset cursor differs from the recorded candidate batch')
        groups = [[Sample.from_dict(item) for item in group] for group in batch['groups']]
        for key, value in batch['after'].items():
            setattr(data_source, key, value)
        if args.rollout_shuffle:
            data_source.dataset.shuffle(data_source.epoch_id)
    else:
        if getattr(data_source, 'buffer', []):
            raise ValueError('The synchronous candidate journal requires an empty replay buffer')
        groups = rollout._pull_sample_groups(data_source, 16)
        batch = {'before': cursor, 'after': {key: getattr(data_source, key) for key in cursor_fields},
                 'groups': [[item.to_dict() for item in group] for group in groups]}
        temporary = batch_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(batch))
        temporary.replace(batch_path)

    async def collect(position, group, client):
        async with semaphore:
            payload = rollout._build_task_payload(args=args, config=config, group=group,
                                                 rollout_id=rollout_id, task_position=position)
            payload["task_id"] = f"clm-{args.clm_run_id}-{rollout_id:05d}-{candidate_batch:03d}-{position:03d}"
            rollout._attach_scheduler_metadata(payload, group_id=group[0].group_index,
                                                policy_version=rollout_id, rollout_step=rollout_id)
            intent = directory / f"group-{position:03d}-intent.json"
            result_path = directory / f"group-{position:03d}-result.json"
            if intent.exists():
                if json.loads(intent.read_text()) != payload:
                    raise ValueError('Existing task intent differs from the frozen candidate')
            else:
                with intent.open('x') as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                response = await client.post(config.rollout_server_url + '/rollout/task/submit', json=payload)
                (directory / f'group-{position:03d}-submit.json').write_text(response.text)
                response.raise_for_status()
            if result_path.exists():
                result = TaskResult.model_validate_json(result_path.read_text())
            else:
                while True:
                    progress = await client.get(config.rollout_server_url + '/rollout/status')
                    progress.raise_for_status()
                    task_status = progress.json()['tasks'].get(payload['task_id'])
                    if task_status is None:
                        raise ValueError('Existing canonical task disappeared; do not resubmit')
                    if task_status != 'running':
                        response = await client.get(config.rollout_server_url + '/rollout/task/' + payload['task_id'])
                        response.raise_for_status()
                        status = TaskStatus.model_validate(response.json())
                        result = TaskResult(task_id=status.task_id, status=status.status,
                                            results=status.results, result_paths=status.result_paths)
                        break
                    await asyncio.sleep(10)
                temporary = result_path.with_suffix('.tmp')
                temporary.write_text(result.model_dump_json(indent=2))
                temporary.replace(result_path)
            try:
                from .native_failures import training_view, restore_native_metadata
                adapter_result, dispositions = training_view(result)
                samples = rollout._convert_task_result_to_samples(
                    config, adapter_result, group, max_tokens=rollout._resolve_max_tokens(args))
                restore_native_metadata(samples, dispositions)
            except (ValueError, TypeError) as error:
                return {'group_index': group[0].group_index, 'samples': [], 'rejection': str(error)}
            expected = {item.session_id: len(item.trajectory.traces) for item in result.results}
            actual = {}
            for sample in samples:
                session = sample.metadata['polar']['session_id']
                actual[session] = actual.get(session, 0) + 1
            rejection = ('Polar adapter dropped or synthesized a recorded model call'
                         if actual != expected else None)
            return {'group_index': group[0].group_index, 'samples': samples, 'rejection': rejection}

    candidates = []
    executed = []
    skipped = []
    eligible_count = 0
    for position, group in enumerate(groups):
        started = (directory / f"group-{position:03d}-intent.json").exists()
        if not should_admit_candidate(eligible_count, required_groups, started):
            skipped.append(position)
            continue
        if not started:
            await wait_for_task_admission(args, rollout_id, candidate_batch, position)
        async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
            candidate = await collect(position, group, client)
        candidates.append(candidate)
        executed.append(position)
        keep, _, _ = candidate_decision(args, candidate)
        eligible_count += int(keep)
    admission = {
        'planned_positions': list(range(len(groups))),
        'executed_positions': executed,
        'unstarted_skipped_positions': skipped,
        'required_eligible_groups': required_groups,
        'eligible_groups': eligible_count,
        'all_started_groups_terminal_before_return': True,
        'selection_order': 'frozen candidate order',
        'cursor_after': batch['after'],
    }
    admission_path = directory / 'ADMISSION_RESULT.json'
    if admission_path.exists() and json.loads(admission_path.read_text()) != admission:
        raise ValueError('Recorded admission differs from completed candidate results')
    temporary = admission_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(admission, indent=2))
    temporary.replace(admission_path)
    return candidates
