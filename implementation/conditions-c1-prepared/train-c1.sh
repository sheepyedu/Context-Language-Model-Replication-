#!/bin/bash
set -euo pipefail
ROOT=/mount-data/CLM-Replication
RUN="$ROOT/runs/c1-seed1234"
mkdir -p "$RUN"
cd "$ROOT/src/slime"
source scripts/models/qwen3.5-9B.sh
export CLM_RESUME_INITIAL_GATEWAY=1
export CLM_RAY_ADDRESS=127.0.0.1:29177
"$ROOT/management/runtime.sh" env CUDA_VISIBLE_DEVICES=4,5 POLAR_DUAL_CHANNEL_W_EFF=0 PYTHONPATH="$ROOT/source-revisions/training-v7:$ROOT/src:$ROOT/src/ProRL/src:$ROOT/src/slime:$ROOT/src/Megatron-LM" python -m clm_replication.train \
  "${MODEL_ARGS[@]}" \
  --custom-config-path "$ROOT/management/conditions-c1-prepared/train-config-c1.yaml" \
  --actor-num-nodes 1 --actor-num-gpus-per-node 2 --num-gpus-per-node 2 --colocate \
  --train-env-vars '{"CUDA_DEVICE_MAX_CONNECTIONS":"1","POLAR_DUAL_CHANNEL_W_EFF":"0"}' \
  --hf-checkpoint /mount-data/Models/Qwen3.5-9B \
  --ref-load "$ROOT/checkpoints/qwen35-9b-initial" \
  --load "$RUN/checkpoints" --save "$RUN/checkpoints" --save-interval 5 \
  --prompt-data "$ROOT/data/train-chat.jsonl" --input-key prompt --label-key label --metadata-key metadata --reward-key score \
  --rollout-shuffle --rollout-seed 1234 --seed 1234 --num-rollout 70 \
  --rollout-batch-size 8 --n-samples-per-prompt 32 --global-batch-size 256 \
  --rollout-max-prompt-len 26624 --rollout-max-response-len 4096 \
  --rollout-temperature 0.7 --rollout-top-p 0.95 \
  --rollout-function-path clm_replication.bridge.generate_rollout \
  --eval-function-path clm_replication.bridge.generate_rollout \
  --custom-reward-post-process-path clm_replication.bridge.post_process_rewards \
  --custom-advantage-function-path clm_replication.advantages.custom_advantage_function \
  --use-dynamic-global-batch-size --num-steps-per-rollout 1 \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
  --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu 32768 \
  --calculate-per-token-loss --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --advantage-estimator grpo --kl-loss-coef 0.01 --use-kl-loss --kl-loss-type low_var_kl \
  --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --use-tis --tis-clip 2 --tis-clip-low 0 \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1 \
  --optimizer-cpu-offload --use-precision-aware-optimizer \
  --no-gradient-accumulation-fusion --bf16 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --rollout-num-gpus-per-engine 2 --sglang-router-ip 127.0.0.1 --sglang-router-port 19174 \
  --sglang-served-model-name qwen35-clm --sglang-context-length 32768 \
  --sglang-mem-fraction-static 0.75 --sglang-disable-cuda-graph --sglang-max-running-requests 4 \
  --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
  --save-debug-rollout-data "$RUN/training/rollout-{rollout_id}.pt" \
  --save-debug-train-data "$RUN/training/train-{rollout_id}.pt" \
  --eval-interval 5 --eval-prompt-data openresearcher "$ROOT/data/validation-chat.jsonl" \
  --n-samples-per-eval-prompt 1 --skip-eval-before-train
