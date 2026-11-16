"""Regress question accuracy, missing outcomes and the frozen one-SE rule."""
from copy import deepcopy
import json
from pathlib import Path
import time

from question_metrics import aggregate_questions, select_one_standard_error, require_distinct_task_namespace


def manifest(ids, step=5, purpose='validation'):
    return {'condition': 'fixture-c2', 'checkpoint_step': step, 'purpose': purpose,
            'dataset_sha256': 'synthetic-fixture', 'terminal_session_statuses': ['COMPLETED', 'ERROR', 'TIMEOUT'],
            'cases': [{'qid': qid, 'task_id': f'task-{index}'} for index, qid in enumerate(ids)]}


def task(task_id, sid, qid, success, traces=1, status='COMPLETED', policy=5):
    return {'task_id': task_id, 'status': 'completed', 'results': [
        {'session_id': sid, 'task_id': task_id, 'status': status,
         'trajectory': {'traces': [{}] * traces, 'metadata': {
             'policy_version': policy,
             'task_metadata': {'qid': qid, 'condition': 'fixture-c2', 'purpose': 'validation'},
             'evaluation': {'outcome_reward': int(success) if type(success) is bool else None,
                            'clm': {'success': success, 'judge': {'success': success}, 'flops': 100}}}}}]}


def reject(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


checks = []
m = manifest(['A', 'B'])
tasks = {'task-0': task('task-0', 'sid-A', 'A', True),
         'task-1': task('task-1', 'sid-B', 'B', False, traces=9)}
summary = aggregate_questions(m, tasks)
assert summary['accuracy'] == .5 and summary['correct_questions'] == 1
assert [row['trajectory_trace_count'] for row in summary['rows']] == [1, 9]
checks.append('two_questions_weight_equally_despite_one_and_nine_traces')
unknown = deepcopy(tasks)
unknown['task-1']['results'][0]['trajectory']['metadata']['evaluation']['clm']['judge']['success'] = None
partial = aggregate_questions(m, unknown)
assert partial['accuracy'] is None and partial['accuracy_bounds'] == [.5, 1]
assert partial['rows'][1]['success'] is None and not partial['eligible_for_selection']
checks.append('unknown_judge_preserves_null_and_full_denominator_bounds')
errored = deepcopy(tasks)
errored['task-1']['results'][0]['status'] = 'ERROR'
assert aggregate_questions(m, errored)['accuracy'] == .5
checks.append('known_negative_terminal_error_is_not_silently_dropped')
typed = manifest([1, '1'])
typed_tasks = {'task-0': task('task-0', 's1', 1, True), 'task-1': task('task-1', 's2', '1', False)}
assert aggregate_questions(typed, typed_tasks)['questions'] == 2
checks.append('original_integer_and_string_identifiers_remain_distinct')
duplicate = manifest(['A', 'A'])
reject(aggregate_questions, duplicate, {})
duplicate = deepcopy(m)
duplicate['cases'][1]['task_id'] = 'task-0'
reject(aggregate_questions, duplicate, {})
checks.append('duplicate_frozen_question_or_task_id_rejected')
bad = deepcopy(tasks)
bad['task-1']['results'][0]['session_id'] = 'sid-A'
reject(aggregate_questions, m, bad)
checks.append('cross_question_session_reuse_rejected')
bad = deepcopy(tasks)
bad['task-1']['results'] *= 2
reject(aggregate_questions, m, bad)
checks.append('multiple_sessions_cannot_be_selected_for_a_one_session_question')
bad = deepcopy(tasks)
bad['task-1']['results'][0]['trajectory']['metadata']['evaluation']['clm']['judge']['success'] = True
reject(aggregate_questions, m, bad)
checks.append('conflicting_evaluator_and_original_judge_rejected')
bad = deepcopy(tasks)
bad['task-1']['results'][0]['trajectory']['metadata']['task_metadata']['qid'] = 'A'
reject(aggregate_questions, m, bad)
checks.append('raw_recorded_question_identity_must_match_manifest')
bad = deepcopy(tasks)
bad['task-1']['task_id'] = 'other'
reject(aggregate_questions, m, bad)
checks.append('raw_response_task_id_must_match_request')
pending = deepcopy(tasks)
pending['task-1']['status'] = 'running'
assert aggregate_questions(m, pending)['accuracy'] is None
assert aggregate_questions(m, {})['known_objective_results'] == 0
checks.append('missing_and_pending_results_are_not_zero_scores')
bad_type = deepcopy(tasks)
bad_type['task-1']['results'][0]['trajectory']['metadata']['evaluation']['clm']['success'] = 0
assert aggregate_questions(m, bad_type)['accuracy'] is None
checks.append('numeric_boolean_coercion_is_not_applied')
costs = deepcopy(tasks)
costs['task-1']['results'][0]['trajectory']['metadata']['evaluation']['clm']['flops'] = float('inf')
assert aggregate_questions(m, costs)['known_costs'] == 1
checks.append('nonfinite_cost_is_missing_without_changing_objective_label')
missing_policy = deepcopy(tasks)
missing_policy['task-0']['results'][0]['trajectory']['metadata'].pop('policy_version')
assert aggregate_questions(m, missing_policy)['accuracy'] is None
wrong_policy = deepcopy(tasks)
wrong_policy['task-0']['results'][0]['trajectory']['metadata']['policy_version'] = 4
reject(aggregate_questions, m, wrong_policy)
checks.append('missing_or_mismatched_policy_cannot_be_assigned_to_checkpoint')
wrong_condition = deepcopy(tasks)
wrong_condition['task-0']['results'][0]['trajectory']['metadata']['task_metadata']['condition'] = 'fixture-c2-seed-other'
reject(aggregate_questions, m, wrong_condition)
checks.append('cross_seed_or_condition_task_results_cannot_be_reused')


def scored_summary(n, correct, step):
    current = manifest(list(range(n)), step)
    results = {f'task-{i}': task(f'task-{i}', f'sid-{i}', i, i < correct, policy=step) for i in range(n)}
    return aggregate_questions(current, results)


rows = [scored_summary(500, 390, 5), scored_summary(500, 392, 10), scored_summary(500, 400, 15)]
chosen = select_one_standard_error(rows)
assert chosen['selected_step'] == 10 and chosen['best_validation_accuracy'] == .8
checks.append('original_500_question_one_SE_rule_selects_earliest_qualifying_checkpoint')
scheduled = select_one_standard_error(rows, [5, 10, 15])
assert scheduled['selected_step'] == 10 and scheduled['selection_is_final']
assert select_one_standard_error(rows, [0, 5, 10, 15])['selected_step'] is None
assert select_one_standard_error(rows)['selection_is_final'] is False
checks.append('full_declared_checkpoint_schedule_required_for_final_selection')
reject(select_one_standard_error, rows, [5, 10])
reject(select_one_standard_error, rows, [5, 5, 10, 15])
checks.append('unplanned_or_duplicate_expected_checkpoint_steps_rejected')
assert select_one_standard_error([scored_summary(4, 1, 5), scored_summary(4, 2, 10)])['selected_step'] == 5
checks.append('exact_one_SE_threshold_is_inclusive')
final = deepcopy(summary)
final['purpose'] = 'final_test'
reject(select_one_standard_error, [final])
reject(select_one_standard_error, [summary, final])
checks.append('final_test_cannot_enter_validation_selection')
mixed = deepcopy(summary)
mixed['condition'] = 'different-condition'
reject(select_one_standard_error, [summary, mixed])
mixed = deepcopy(summary)
mixed['dataset_sha256'] = 'different-dataset'
reject(select_one_standard_error, [summary, mixed])
checks.append('conditions_and_frozen_datasets_are_not_pooled')
assert select_one_standard_error([partial])['selected_step'] is None
checks.append('incomplete_validation_never_silently_selects_a_checkpoint')
bad = deepcopy(summary)
bad['accuracy'] = .9
reject(select_one_standard_error, [bad])
reject(select_one_standard_error, [summary, summary])
checks.append('count_inconsistent_accuracy_and_duplicate_checkpoint_steps_rejected')
other = deepcopy(m)
other['condition'] = 'other-seed'
reject(require_distinct_task_namespace, other, m)
other['cases'] = [{'qid': case['qid'], 'task_id': 'other-' + case['task_id']} for case in other['cases']]
require_distinct_task_namespace(other, m)
require_distinct_task_namespace(m, m)
checks.append('shared_server_task_cache_namespace_checked_without_identifier_rewriting')
proof = {'accepted': True, 'created': time.time(), 'checks': checks, 'count': len(checks),
         'synthetic_data_only': True, 'actual_model_HTTP_or_GPU_calls': 0,
         'actual_500_question_validation_pending': True}
Path(__file__).with_name('CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
