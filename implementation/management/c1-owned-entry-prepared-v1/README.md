# C1 owned-entry preparation

This pending control uses the accepted task-only TP4 configuration in `conditions-c1-tp4-v2`. It is prepared independently while C2 continues on physical GPUs4/5/6/7.

## Controller and admission

`prepare_owner.py` derives `owned_c1.py` from the accepted C2 v14 controller. Only the run directory and training-script path change. Router startup, bounded readiness, native child wait and integer exit collection remain identical. The C1 training script initializes the original model and its own run/checkpoint/audit paths. It does not point at C2 rollout archives.

`launch_c1.py` first reads the actual preceding C2 pointer, original process-start identity and integer exit. A present original process, including a zombie awaiting reaping, prevents admission. PID reuse is distinguished by the original start tick. A per-run native file lock serializes C1 dispatch attempts. Existing C1 intents require original closed exits and gone owners. Each new intent uses exclusive creation and CURRENT_OWNER publication is atomic.

Formal admission also checks frozen source/PINS, the four authorized GPU UUIDs and absence of compute applications, router-port availability, zero gateway inflight and coordinator pending sessions, terminal native session states, and the original owned Ray session/mapping. These resource and full-dispatch branches remain pending actual admission. They cannot be declared accepted from the predecessor-only CPU tests.

## Current evidence

Ten predecessor/lock CPU checks pass locally and on the server. The actual main CLI was invoked once after the final source change with the preceding C2 owner still live. It refused with integer exit1 before source/network/GPU checks or dispatch. The C1 run tree stayed unchanged and dispatch intents remained zero. This is an expected admission-test refusal, with its original stdout/stderr saved. There were no model, Judge or GPU operations.

The original C1 configuration proof remains separate. It verifies nine effective-argument differences, task-only efficiency weight0, TP4 and a CPU Ray environment override. Actual C1 GPU-role environment, first model consumer, optimizer, checkpoint and full70-update training remain pending.

## Eventual use

After the preceding owner and GPU work are actually closed, invoke through the project runtime so native Polar imports are available.

```sh
/mount-data/CLM-Replication/management/runtime-gpu4567.sh python /mount-data/CLM-Replication/management/c1-owned-entry-prepared-v1/launch_c1.py --attempt 1
```

Reconcile original intent, launch and actual effect before any retry after a disconnected call. Keep all source-stage failures and costs. A changed final C2 configuration requires an explicit matched C1 preparation revision before dispatch. Current C2 is allowed to finish its first actual update with its original accepted profile.
