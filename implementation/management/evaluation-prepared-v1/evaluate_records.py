"""Create immutable question-level reports from saved native task responses."""
import argparse
import hashlib
import json
from pathlib import Path

from question_metrics import aggregate_questions, select_one_standard_error


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    aggregate = commands.add_parser('aggregate')
    aggregate.add_argument('--manifest', type=Path, required=True)
    aggregate.add_argument('--task-results', type=Path, required=True)
    aggregate.add_argument('--output', type=Path, required=True)
    select = commands.add_parser('select')
    select.add_argument('--summaries', type=Path, nargs='+', required=True)
    select.add_argument('--expected-steps', type=int, nargs='+', required=True)
    select.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'aggregate':
        manifest_raw = args.manifest.read_bytes()
        task_raw = args.task_results.read_bytes()
        index = json.loads(task_raw)
        if index['manifest_sha256'] != hashlib.sha256(manifest_raw).hexdigest():
            raise ValueError('Captured tasks belong to a different frozen manifest')
        tasks = {}
        for task_id, source in index['responses'].items():
            raw = Path(source['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source['sha256']:
                raise ValueError('Original task response differs from its saved receipt')
            tasks[task_id] = json.loads(raw)
        report = aggregate_questions(json.loads(manifest_raw), tasks)
        report['source_manifest_sha256'] = hashlib.sha256(manifest_raw).hexdigest()
        report['source_task_results_sha256'] = hashlib.sha256(task_raw).hexdigest()
    else:
        summaries = [json.loads(path.read_bytes()) for path in args.summaries]
        report = select_one_standard_error(summaries, args.expected_steps)
        report['source_summary_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in args.summaries}
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
