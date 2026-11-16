# Adjacent-update acceptance preparation

This one-shot management helper prepares evidence for consecutive real updates. It reuses the pinned first-update helper for native finite loss/gradient parsing, exact owner identity and finite language-tensor changes. It does not enter the training source or change the actor schedule.

## Order

1. Wait for the actual first optimizer update, raw serving version2 and successful `first_update_acceptance_v4.py`. The existing initial snapshot stays in place throughout that acceptance.
2. Invoke this helper for logical policy1. It requires the accepted first-update record and exact current owner. It then snapshots the current synchronized weights through the native checker and immediately compares against that snapshot. The native snapshot is a CPU diagnostic copy. Model parameters are not written.
3. At logical policy2, require raw version3 and the accepted policy1 anchor. Compare current weights with that previous anchor, retain actual positive finite language changes, then establish the next snapshot. Repeat for policy3/raw4.

The policy step numbering is derived from the accepted first native metric. Duplicate identical metric rows count once. Missing steps, conflicting duplicates, changed owners, nonadjacent versions, invalid model identity and missing prior anchors are rejected before a new diagnostic intent.

Each policy and each diagnostic action has an exclusive intent. Raw request/response, HTTP status, partial data and parsed metrics are retained. A transport error leaves an unknown diagnostic outcome. Inspect that original intent and effect before any recovery. The helper does not replay an existing action, reset tensors, pause training or create another observer.

## Invocation after the required real evidence exists

```sh
/mount-data/CLM-Replication/env/rl/bin/python /mount-data/CLM-Replication/management/update-chain-prepared-v1/accept_update_chain.py --policy 1 --attempt 13 --owner-start 51919204
```

The actual owner, attempt and start must be read from current original records at use time. Use policy2 and3 only in their corresponding actual synchronized versions. If an owner or version boundary is missed, preserve the incomplete evidence and reconcile the chain without claiming consecutive verification.

## Scope

Fourteen CPU checks pass, including thirteen invalid prefix cases, the actual pinned finite-metric parser, durable unknown/error records, mutation-action rejection and a complete two-update file/protocol fixture with fake native responses. These tests make zero real HTTP, GPU, model or Judge calls. The real first update and adjacent comparisons remain pending.

The helper accepts only parameter and native-metric evidence. Its result explicitly leaves `stability_complete=false`. Actual mask, reward, accounting and checkpoint evidence are evaluated separately before declaring training stable or completing subtask2. A native snapshot is a diagnostic baseline and does not establish disk checkpoint restore.
