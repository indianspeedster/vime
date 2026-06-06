#!/bin/bash

set -ex
source ~/.bashrc
ulimit -n 65535

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export PYTHONBUFFERED=16
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_DISABLE_SIGINT_OVERRIDE=1
export HCCL_CONNECT_TIMEOUT=7200

export PYTHONPATH="/home/ma-user/Megatron-LM:/home/ma-user/vllm:/home/ma-user/vime:${PYTHONPATH}"

source "${SCRIPT_DIR}/models/qwen3-32B.sh"

CKPT_ARGS=(
   --hf-checkpoint /data/local_models/Qwen3-32B
   --ref-load /data/local_models/Qwen3-32B_torch_dist
)

PROMPT_SET=/data/nfs_87/xky/datasets/gsm8k/train.parquet

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key question
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 20
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --vllm-max-model-len 2048
   --vllm-gpu-memory-utilization 0.60
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --no-gradient-accumulation-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 2
   --vllm-enforce-eager
)

MISC_ARGS=(
   --transformer-impl local
   --seq-length 2048
   --qkv-format bshd
   --micro-batch-size 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
   --no-persist-layer-norm
   --no-masked-softmax-fusion
   --no-bias-dropout-fusion
   --make-vocab-size-divisible-by 1
   --max-num-steps 300
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 0 --resources '{"NPU": 16}' --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/home/ma-user/Megatron-LM:/home/ma-user/vllm:/home/ma-user/vime:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:'"$PYTHONPATH"'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
    "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest/",
    "ASCEND_AICPU_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
    "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest/",
    "HYDRA_FULL_ERROR": "1",
    "RAY_DEBUG_POST_MORTEM_DISABLED": "1",
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann-8.5.2/lib64:'"$LD_LIBRARY_PATH"'"
  }
}'

cd /home/ma-user/vime
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /home/ma-user/vime/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
