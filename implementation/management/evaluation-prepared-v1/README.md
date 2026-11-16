# Question-level validation and checkpoint selection

This independent analysis path implements the frozen plan's validation accuracy over500 original questions and its one-binomial-standard-error checkpoint rule. These tools operate on saved native task responses as an independent analysis step.

## Why this correction is required

The original native eval bridge expands a completed trajectory into per-call samples. Its default rewards array therefore repeats a task reward for every model call. A native CPU fixture with one successful one-call question and one failed nine-call question produces array mean0.1. Question accuracy is0.5. The native completed-session auxiliary metric is already deduplicated, while the default missing-reward extractor returns0. These diagnostics cannot supply the plan's full500-question objective accuracy by themselves.

`question_metrics.py` reads each original TaskStatus/SessionResult once per frozen question. It checks exact task/session IDs, raw typed question IDs, condition, purpose and policy version. Evaluator and saved judge booleans must agree. Known terminal errors retain their actual objective label. Missing, pending or ungraded results retain null. Accuracy over the full dataset exists only with complete objective coverage. Partial reports include known-subset accuracy and full-denominator bounds. Objective accuracy, task reward and FLOPs have separate coverage fields.

Model selection uses the highest full-dataset validation accuracy p, computes sqrt(p*(1-p)/N), and selects the earliest checkpoint at or above p-SE. Conditions and datasets stay separate. Final-test data are refused. A final selection also requires the complete declared checkpoint schedule and complete objective coverage at every scheduled point.

## Components

- `compile_manifest.py` uses the actual native task renderer and eval ID producer. It freezes query IDs, task IDs, condition, purpose, dataset SHA and native terminal states. Its records contain the original query IDs and native request identities. `EXPECTED_C2_POLICY5_VALIDATION.json` prepares500 expected identities and contains no model results.
- `collect_task_results.py` performs bounded read-only coordinator GET batches, with a35-second process alarm and exclusive export lock. It preserves raw response bytes, partial/error receipts and immutable content-addressed indices. A repeat batch reuses already closed captures. Its transport is coordinator GET.
- `evaluate_records.py aggregate` reads a hash-bound capture index and produces an immutable per-question report. `select` applies the declared validation schedule. Use a new output artifact name when the underlying evidence has actually changed.

## Eventual use after real validation

```sh
ROOT=/mount-data/CLM-Replication
PREP=$ROOT/management/evaluation-prepared-v1
OUT=$ROOT/runs/c2-seed1234/evaluation/policy-00005
mkdir -p "$OUT"
"$ROOT/management/retrieval-runtime.sh" "$PREP/compile_manifest.py" --config "$ROOT/management/train-config-v7.yaml" --data "$ROOT/data/validation-chat.jsonl" --dataset-name openresearcher --checkpoint-step 5 --purpose validation --expected-count 500 --output "$OUT/expected.json"
python3 "$PREP/collect_task_results.py" --manifest "$OUT/expected.json" --directory "$OUT/captures"
python3 "$PREP/evaluate_records.py" aggregate --manifest "$OUT/expected.json" --task-results "$OUT/captures/TASK_RESULTS_INDEX.json" --output "$OUT/summary.json"
```

The collector prints the immutable index path for exact audit reuse. More bounded export batches can finish an incomplete capture without repeating any policy generation or Judge request. Retain original native diagnostics alongside these reports. Keep real response files and session identifiers in the private CPFS artifact directory.

Final planned selection uses every scheduled point0,5,...,70.

```python
from pathlib import Path
import subprocess
import sys
root = Path('/mount-data/CLM-Replication')
run = root / 'runs/c2-seed1234'
steps = list(range(0, 71, 5))
summaries = [str(run / f'evaluation/policy-{step:05d}/summary.json') for step in steps]
subprocess.run([sys.executable, str(root / 'management/evaluation-prepared-v1/evaluate_records.py'),
                'select', '--summaries', *summaries, '--expected-steps', *map(str, steps),
                '--output', str(run / 'evaluation/SELECTION.json')], check=True)
```

## Remaining experiment obligations

Current training passes `--skip-eval-before-train`. The planned initial step0 measurement is therefore deferred. It still needs an actual initial-model/baseline validation record before final selection. This temporal deviation must be disclosed. A compiled step0 identity manifest would not supply that result.

The legacy template has a second issue for future seeds. Changing clm_run_id alone leaves eval task IDs unchanged. The native CPU producer reproduces this collision. Unstarted run configurations must include their full run identity in the task namespace, for example `clm-{args.clm_run_id}-eval-{rollout_id}-{sample.metadata.qid}`, and check against prior manifests with `--avoid-manifests`. Current in-flight task identities are preserved. The aggregator also refuses cross-condition evidence.

## Evidence and limits

24 pure CPU checks,8 actual native-producer/schema CPU checks and9 collector/CLI CPU checks pass. Native submit and trace conversion are explicitly stubbed. Original one-call/nine-call behavior is separately reproduced in `NATIVE_METRIC_UNIT_ACTUAL.json`. A failed native-schema fixture used lowercase trajectory status. Its original helper/disposition remain. The correction only changes that synthetic fixture to native uppercase terminal status.

Actual500-question validation, checkpoint selection and830-question final test remain pending. Preparation used CPU fixtures with actual model/Judge/GPU call counts of0. The PPO run continues on its frozen profile.
