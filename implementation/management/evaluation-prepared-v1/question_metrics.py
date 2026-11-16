"""Aggregate original single-session task outcomes at the question level."""
import hashlib
import json
import math


def question_id_key(value):
    if type(value) not in (str, int):
        raise ValueError('Question identifiers must preserve their original string or integer type')
    return type(value).__name__, value


def finite_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return value


def require_distinct_task_namespace(current, previous):
    fields = ['condition', 'purpose', 'checkpoint_step', 'dataset_sha256']
    if all(current[field] == previous[field] for field in fields):
        return
    current_ids = {case['task_id'] for case in current['cases']}
    previous_ids = {case['task_id'] for case in previous['cases']}
    if current_ids & previous_ids:
        raise ValueError('Task identities collide across distinct frozen evaluations')


def aggregate_questions(manifest, tasks):
    cases = manifest['cases']
    if not cases:
        raise ValueError('An empty dataset cannot define question accuracy')
    ids = [question_id_key(case['qid']) for case in cases]
    task_ids = [case['task_id'] for case in cases]
    if len(set(ids)) != len(ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError('Frozen questions and task IDs must each be unique')
    if any(type(value) is not str or not value for value in task_ids):
        raise ValueError('Task IDs must be nonempty original strings')
    unexpected = set(tasks) - set(task_ids)
    if unexpected:
        raise ValueError('Task responses include IDs outside the frozen dataset')
    terminal = set(manifest['terminal_session_statuses'])
    seen_sessions = set()
    rows = []
    for case in cases:
        row = {'qid': case['qid'], 'task_id': case['task_id'], 'session_id': None,
               'task_status': None, 'session_status': None, 'success': None,
               'policy_version': None,
               'task_reward': None, 'flops': None, 'trajectory_trace_count': None,
               'disposition': 'missing_task_result'}
        rows.append(row)
        task = tasks.get(case['task_id'])
        if task is None:
            continue
        if task['task_id'] != case['task_id']:
            raise ValueError('Response task identity differs from the frozen request')
        row['task_status'] = task['status']
        if task['status'] not in {'completed', 'failed'}:
            row['disposition'] = 'pending_task'
            continue
        sessions = task.get('results', [])
        if not sessions:
            row['disposition'] = 'missing_session_result'
            continue
        if len(sessions) != 1:
            raise ValueError('The evaluation contract requires one original session per question')
        session = sessions[0]
        if session['task_id'] != case['task_id']:
            raise ValueError('Session belongs to a different task')
        sid = session['session_id']
        if type(sid) is not str or not sid or sid in seen_sessions:
            raise ValueError('Session IDs must be nonempty and unique across questions')
        seen_sessions.add(sid)
        row['session_id'] = sid
        row['session_status'] = session['status']
        if session['status'] not in terminal:
            row['disposition'] = 'pending_session'
            continue
        trajectory = session.get('trajectory') or {}
        metadata = trajectory.get('metadata') or {}
        task_metadata = metadata.get('task_metadata') or {}
        if task_metadata.get('condition') != manifest['condition'] or task_metadata.get('purpose') != manifest['purpose']:
            raise ValueError('Recorded condition or purpose differs from the frozen experiment')
        version = metadata.get('policy_version')
        row['policy_version'] = version
        if version is None:
            row['disposition'] = 'unknown_policy_identity'
            continue
        if type(version) is not int or version != manifest['checkpoint_step']:
            raise ValueError('Recorded policy version differs from the frozen checkpoint')
        recorded_qid = task_metadata.get('qid')
        if recorded_qid is not None and question_id_key(recorded_qid) != question_id_key(case['qid']):
            raise ValueError('Recorded raw question ID differs from the frozen request')
        evaluation = metadata.get('evaluation') or {}
        outcome = evaluation.get('clm') or {}
        judge = outcome.get('judge') or {}
        success = outcome.get('success')
        judged_success = judge.get('success')
        row['task_reward'] = finite_number(evaluation.get('outcome_reward'))
        cost = finite_number(outcome.get('flops'))
        row['flops'] = cost if cost is not None and cost >= 0 else None
        traces = trajectory.get('traces')
        row['trajectory_trace_count'] = len(traces) if isinstance(traces, list) else None
        if type(success) is not bool or type(judged_success) is not bool:
            row['disposition'] = 'unknown_objective_result'
            continue
        if success != judged_success:
            raise ValueError('Evaluator outcome and saved judge result conflict')
        row['success'] = success
        row['disposition'] = 'scored'
    known = [row for row in rows if row['success'] is not None]
    correct = sum(row['success'] for row in known)
    total = len(rows)
    unknown = total - len(known)
    rewards = [row['task_reward'] for row in rows if row['task_reward'] is not None]
    costs = [row['flops'] for row in rows if row['flops'] is not None]
    identity = json.dumps(sorted(ids), ensure_ascii=False, separators=(',', ':')).encode()
    return {
        'condition': manifest['condition'], 'checkpoint_step': manifest['checkpoint_step'],
        'purpose': manifest['purpose'],
        'dataset_sha256': manifest['dataset_sha256'],
        'question_set_sha256': hashlib.sha256(identity).hexdigest(),
        'questions': total, 'known_objective_results': len(known), 'unknown_objective_results': unknown,
        'correct_questions': correct, 'accuracy': correct / total if unknown == 0 else None,
        'known_subset_accuracy': correct / len(known) if known else None,
        'accuracy_bounds': [correct / total, (correct + unknown) / total],
        'eligible_for_selection': unknown == 0,
        'known_task_rewards': len(rewards), 'mean_known_task_reward': sum(rewards) / len(rewards) if rewards else None,
        'known_costs': len(costs), 'mean_known_flops': sum(costs) / len(costs) if costs else None,
        'rows': rows,
        'scope': 'one saved objective result per original question, diagnostic call-level metrics retained separately',
    }


def select_one_standard_error(summaries, expected_steps=None):
    if not summaries:
        raise ValueError('No validation summaries were supplied')
    first = summaries[0]
    if first['purpose'] != 'validation':
        raise ValueError('Model selection requires validation outcomes')
    signature = (first['condition'], first['dataset_sha256'], first['question_set_sha256'], first['questions'])
    steps = set()
    eligible = []
    for summary in summaries:
        if summary['purpose'] != 'validation':
            raise ValueError('Final-test outcomes cannot enter model selection')
        actual = (summary['condition'], summary['dataset_sha256'], summary['question_set_sha256'], summary['questions'])
        if actual != signature:
            raise ValueError('Checkpoint selection must use the same condition and frozen questions')
        step = summary['checkpoint_step']
        if type(step) is not int or step < 0 or step in steps:
            raise ValueError('Checkpoint steps must be unique nonnegative integers')
        steps.add(step)
        known = summary['known_objective_results']
        total = summary['questions']
        correct = summary['correct_questions']
        if type(total) is not int or total <= 0 or type(known) is not int or not 0 <= known <= total:
            raise ValueError('Invalid objective-result coverage')
        if type(correct) is not int or not 0 <= correct <= known:
            raise ValueError('Invalid correct-question count')
        complete = known == total
        if summary['eligible_for_selection'] is not complete:
            raise ValueError('Selection eligibility differs from objective coverage')
        if not complete:
            if summary['accuracy'] is not None:
                raise ValueError('Incomplete coverage cannot have a full-dataset point accuracy')
            continue
        accuracy = summary['accuracy']
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or accuracy != correct / total:
            raise ValueError('Accuracy differs from the original question counts')
        eligible.append(summary)
    if expected_steps is not None:
        if (not expected_steps or any(type(step) is not int or step < 0 for step in expected_steps)
                or len(set(expected_steps)) != len(expected_steps)):
            raise ValueError('Expected checkpoint schedule is invalid')
        missing = set(expected_steps) - steps
        unexpected = steps - set(expected_steps)
        if unexpected:
            raise ValueError('Checkpoint summaries include steps outside the planned schedule')
        if missing:
            return {'selected_step': None, 'reason': 'incomplete_checkpoint_schedule',
                    'missing_steps': sorted(missing), 'selection_is_final': False}
    if not eligible:
        return {'selected_step': None, 'reason': 'no_complete_validation', 'eligible_checkpoints': 0,
                'selection_is_final': False}
    best = max(eligible, key=lambda summary: summary['accuracy'])
    standard_error = math.sqrt(best['accuracy'] * (1 - best['accuracy']) / best['questions'])
    threshold = best['accuracy'] - standard_error
    selected = min((summary for summary in eligible if summary['accuracy'] >= threshold),
                   key=lambda summary: summary['checkpoint_step'])
    return {'selected_step': selected['checkpoint_step'], 'best_validation_accuracy': best['accuracy'],
            'binomial_standard_error': standard_error, 'selection_threshold': threshold,
            'selected_accuracy': selected['accuracy'], 'questions': best['questions'],
            'eligible_checkpoints': len(eligible), 'incomplete_checkpoints': len(summaries) - len(eligible),
            'selection_is_final': expected_steps is not None and len(eligible) == len(summaries),
            'rule': 'earliest checkpoint within one binomial standard error of the best full-dataset accuracy'}
