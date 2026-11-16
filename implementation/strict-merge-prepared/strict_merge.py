"""Merge training segments only when every original causal input is identical."""
from copy import deepcopy

REQUIRED = {'tokens', 'response_lengths', 'rewards', 'raw_reward', 'truncated',
            'sample_indices', 'loss_masks', 'rollout_log_probs', 'a_eff', 'role_masks'}
RESPONSE_FIELDS = ('loss_masks', 'rollout_log_probs', 'role_masks')
SEQUENCE_FIELDS = ('rewards', 'raw_reward', 'truncated', 'a_eff')


def merge_train_data(data, identities, max_tokens=32768):
    if set(data) != REQUIRED:
        raise ValueError('Strict text merge requires the explicitly validated training fields')
    count = len(data['tokens'])
    if len(identities) != count or any(len(values) != count for values in data.values()):
        raise ValueError('Training fields and source identities differ in length')
    output = {name: [] for name in data}
    spans = []
    for index in range(count):
        tokens = data['tokens'][index]
        response_length = data['response_lengths'][index]
        prompt_length = len(tokens) - response_length
        if not 0 < response_length < len(tokens) <= max_tokens:
            raise ValueError('Invalid source context or response length')
        if any(len(data[name][index]) != response_length for name in RESPONSE_FIELDS):
            raise ValueError('Original response masks/logprobs are incomplete')
        identity = identities[index]
        if set(identity) != {'session_id', 'group_index', 'policy_version', 'trace_index'}:
            raise ValueError('Original source identity is incomplete')
        same_session = bool(spans) and all(
            spans[-1][-1]['identity'][key] == identity[key]
            for key in ('session_id', 'group_index', 'policy_version'))
        adjacent = same_session and identity['trace_index'] == spans[-1][-1]['identity']['trace_index'] + 1
        prefix = bool(output['tokens']) and tokens[:prompt_length][:len(output['tokens'][-1])] == output['tokens'][-1]
        same_sequence_values = bool(spans) and all(output[name][-1] == data[name][index] for name in SEQUENCE_FIELDS)
        join = adjacent and prefix and same_sequence_values
        if join:
            old_length = len(output['tokens'][-1])
            gap = prompt_length - old_length
            assert gap >= 0
            output['tokens'][-1].extend(tokens[old_length:])
            output['response_lengths'][-1] += gap + response_length
            for name in RESPONSE_FIELDS:
                output[name][-1].extend([0] * gap + list(data[name][index]))
        else:
            for name in data:
                output[name].append(deepcopy(data[name][index]))
            spans.append([])
        spans[-1].append({'identity': dict(identity), 'source_index': index,
                         'generated_start': prompt_length, 'generated_length': response_length})
    # Every generated position is trained under its exact original prefix.
    for merged_index, sources in enumerate(spans):
        merged = output['tokens'][merged_index]
        offset = len(merged) - output['response_lengths'][merged_index]
        for source in sources:
            index = source['source_index'];start = source['generated_start'];size = source['generated_length']
            assert merged[:start + size] == data['tokens'][index]
            for name in RESPONSE_FIELDS:
                assert output[name][merged_index][start - offset:start - offset + size] == data[name][index]
    assert sum(map(sum, output['loss_masks'])) == sum(map(sum, data['loss_masks']))
    assert sum(map(sum, output['role_masks'])) == sum(map(sum, data['role_masks']))
    return output, spans
