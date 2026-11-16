"""Prepare equivalent text training segments before native batch sizing.

This candidate remains separate from the active training source.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .strict_merge import merge_train_data


def prepared_digest(sample):
    """Bind the fields consumed by the standard converter and advantage hook."""
    polar = sample.metadata['polar']
    fields = {'tokens': sample.tokens, 'response_length': sample.response_length,
              'loss_mask': sample.loss_mask, 'rollout_log_probs': sample.rollout_log_probs,
              'role_mask': polar['role_mask'], 'a_eff': sample.metadata['a_eff'],
              'sources': sample.metadata['clm_strict_merge']['sources'],
              'reward': sample.metadata['clm_strict_merge']['raw_reward'],
              'advantage': sample.metadata['clm_strict_merge']['advantage'],
              'group_index': sample.group_index,
              'policy_version': polar['trajectory_metadata']['policy_version'],
              'session_id': polar['session_id']}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def merge_prepared_samples(samples, raw_rewards, advantages, max_tokens=32768):
    """Keep each supervised token under its original complete causal prefix."""
    if not (len(samples) == len(raw_rewards) == len(advantages)):
        raise ValueError('Original rewards and samples have different lengths')
    if any(sample.remove_sample or sample.multimodal_train_inputs is not None
           or sample.rollout_routed_experts is not None or sample.train_metadata is not None
           or sample.teacher_log_probs is not None for sample in samples):
        raise ValueError('This candidate supports complete text-only samples')
    data = {name: [] for name in ('tokens', 'response_lengths', 'rewards', 'raw_reward',
            'truncated', 'sample_indices', 'loss_masks', 'rollout_log_probs', 'a_eff', 'role_masks')}
    identities = []
    for sample, reward, advantage in zip(samples, raw_rewards, advantages, strict=True):
        polar = sample.metadata['polar']
        row = {'tokens': sample.tokens, 'response_lengths': sample.response_length,
               'rewards': advantage, 'raw_reward': reward,
               'truncated': int(sample.status == sample.Status.TRUNCATED),
               'sample_indices': sample.index, 'loss_masks': sample.loss_mask,
               'rollout_log_probs': sample.rollout_log_probs,
               'a_eff': sample.metadata['a_eff'], 'role_masks': polar['role_mask']}
        for name, value in row.items():
            data[name].append(value)
        identities.append({'session_id': polar['session_id'], 'group_index': sample.group_index,
                           'policy_version': polar['trajectory_metadata']['policy_version'],
                           'trace_index': polar['trace_index']})
    merged_data, spans = merge_train_data(data, identities, max_tokens=max_tokens)
    result = []
    for position, sources in enumerate(spans):
        sample = deepcopy(samples[sources[0]['source_index']])
        # The segment is a derived training view. Display text stays in source traces.
        if len(sources) > 1:
            sample.response = ''
        sample.tokens = merged_data['tokens'][position]
        sample.response_length = merged_data['response_lengths'][position]
        sample.loss_mask = merged_data['loss_masks'][position]
        sample.rollout_log_probs = merged_data['rollout_log_probs'][position]
        sample.metadata['polar']['role_mask'] = merged_data['role_masks'][position]
        sample.metadata['clm_strict_merge'] = {
            'sources': sources, 'raw_reward': merged_data['raw_reward'][position],
            'advantage': merged_data['rewards'][position], 'version': 1}
        sample.metadata['clm_strict_merge']['training_sha256'] = prepared_digest(sample)
        result.append(sample)
    before = {(value['group_index'], value['session_id']) for value in identities}
    after = {(sample.group_index, sample.metadata['polar']['session_id']) for sample in result}
    if before != after:
        raise ValueError('Merging changed the original trajectory coverage')
    return result, spans


def post_process_rewards(args, samples):
    """Preserve the original trajectory-level rewards and token role masks."""
    from . import bridge
    markers = ['clm_strict_merge' in (sample.metadata or {}) for sample in samples]
    if not any(markers):
        return bridge.post_process_rewards(args, samples)
    if not all(markers):
        raise ValueError('Mixed prepared and unprepared training samples')
    raw, advantages = [], []
    for sample in samples:
        prepared = sample.metadata['clm_strict_merge']
        if prepared['training_sha256'] != prepared_digest(sample):
            raise ValueError('Prepared training fields changed before native conversion')
        raw.append(prepared['raw_reward'])
        advantages.append(prepared['advantage'])
    return raw, advantages


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Merge only after complete original selection and before native sizing."""
    from . import bridge
    from slime.rollout.base_types import RolloutFnTrainOutput
    output = bridge.generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    if evaluation:
        return output
    samples = output.samples
    while isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]
    raw, advantages = bridge.post_process_rewards(args, samples)
    merged, spans = merge_prepared_samples(samples, raw, advantages,
                                           max_tokens=args.sglang_context_length)
    summary = {'policy_version': rollout_id, 'original_samples': len(samples),
               'training_segments': len(merged),
               'original_token_occurrences': sum(len(sample.tokens) for sample in samples),
               'merged_token_occurrences': sum(len(sample.tokens) for sample in merged),
               'original_trainable_tokens': sum(sum(sample.loss_mask) for sample in samples),
               'merged_trainable_tokens': sum(sum(sample.loss_mask) for sample in merged),
               'source_spans': spans, 'new_generation_calls': 0}
    target = Path(args.clm_audit_dir) / f'strict-merge-policy-{rollout_id:05d}.json'
    if target.exists() and json.loads(target.read_text()) != summary:
        raise ValueError('A previous strict-merge view has different provenance')
    target.write_text(json.dumps(summary, indent=2))
    return RolloutFnTrainOutput(samples=merged, metrics=output.metrics)


def update_policy_version(args, version):
    from .bridge import update_policy_version as original
    return original(args, version)
