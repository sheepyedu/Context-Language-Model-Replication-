# BrowseComp-Plus final-test driver preparation

This candidate completes the driver metadata and task-visible retrieval instructions for the prepared fixed-corpus gateway. It is isolated from current C2 training and has no model or Judge calls.

## Data and visibility

The original 830 query/label records retain SHA `bd1f496296899971f345cb2d4dfa766ac70cfbab243944c493a6527193292b96`. No rows or test evidence documents are changed or added. Labels enter the evaluator configuration. They are absent from the agent instruction and AgentSpec. CPU fixtures use a generic synthetic question and label.

The native task template supplies dataset `browsecomp_plus` and retrieval profile `browsecomp_plus_fixed` in driver-owned session metadata. Matching AgentSpec settings select a fixed-corpus notice. This notice explains the existing native web CLI, top10 search results, 512-Qwen-token snippets and 8192-Qwen-token document reads. The public-web profile preserves the original model-visible instruction bytes.

## Preparation

`prepare_bcp_driver.py` clones the independently accepted gateway candidate, changes only the harness retrieval notice selection and derives a final-test YAML from the original runtime configuration. The original task image, timeout, answer artifact, native CLM runner and evaluator remain. The read-only agent package mount points at this candidate. The original active gateway, trainer and task bundles are untouched.

`check_bcp_driver_cpu.py` executes the actual native config resolver and task renderer, AgentSpec and ClmHarness. It materializes the generated instruction through five real Python subprocesses, all integer exit0. Fourteen checks pass, including exact question preservation, private-label separation, native metadata routing, fixed tool description, original public model-input equality and unchanged runtime budgets.

The first CPU check compared different Python quote spellings in generated shell commands and exited1. Its helper and disposition remain. The corrected check executes both generated scripts and compares actual model-visible instruction bytes. Candidate source was unchanged by this test correction.

## Remaining acceptance

This package is prepared only. Production activation, live coordinator metadata propagation, actual task-container mounts and an actual model consumer still require acceptance before all830 final-test tasks run. Original gateway transport acceptance covers real CLI/registry/corpus HTTP. This driver proof covers native payload and instruction construction. It does not constitute a completed benchmark evaluation.

The final checkpoint must be selected on the original validation set before final test. Per-query gold documents, test grades and Judge feedback cannot enter the policy. Full original rollout/Judge accounting remains required. C1/Summary condition drivers must preserve their own harness and policy identity when selecting the same corpus profile.
