#!/usr/bin/env bash
# =============================================================================
# run_test.sh — Reproducible VIME ROCm test runner (no Buildkite required)
#
# Usage:
#   ./run_test.sh                              # run the 11 known-passing tests
#   ./run_test.sh test_qwen2.5_0.5B_fully_async_short.py  # single test
#   ./run_test.sh --suite short                # short smoke suite
#   ./run_test.sh --suite all                  # all 24 GPU tests
#
# Requirements:
#   - AMD GPU with ROCm (MI300X / MI350X / gfx942+)
#   - Docker with GPU access (--device=/dev/kfd --device=/dev/dri)
#   - ~200 GB disk for models + HF cache
#
# Quick start:
#   1. docker pull vllm/vime-rocm:latest
#   2. export HF_HOME=/path/to/huggingface_cache  # default: ~/.cache/huggingface
#   3. export MODELS_DIR=/path/to/converted_models # default: $HOME/models
#   4. ./run_test.sh
#
# ⚠️  IMPORTANT — ref-load checkpoint conversion:
#   Tests that use --ref-load need a Megatron torch_dist checkpoint, not just
#   the raw HF model. On ROCm the bridge path doesn't work, so you must pre-
#   convert models before running tests. See the "Model Prep" section below
#   or the "Reproducibility Guide" at:
#   https://app.notion.com/p/VIME-ROCm-GPU-CI-Reproducibility-Guide-3a074178c99781de919ce065cd32919d
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ── Config ────────────────────────────────────────────────────────────────
VIME_ROCM_IMAGE="${VIME_ROCM_IMAGE:-vllm/vime-rocm:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
CONTAINER_NAME="vime-test-$(date +%Y%m%d-%H%M%S)"
NUM_GPUS_DEFAULT="${NUM_GPUS:-1}"

# ── All 24 tests (from the mi350-do-01 full suite run) ────────────────────
# Format: "test_file.py:num_gpus:extra_args"
# Status: ✅ = passes, ❌ = known failure (see notes at bottom)

# Group 1: Qwen2.5-0.5B (fast, ~2–5 min each)
Q25_TESTS=(
    "test_qwen2.5_0.5B_fully_async_short.py:4"       # ❌  needs ref-load
    "test_qwen2.5_0.5B_async_short.py:4"              # ❌  needs ref-load
    "test_qwen2.5_0.5B_short.py:4"                    # ❌  needs ref-load
    "test_qwen2.5_0.5B_fanout_short.py:4"             # ❌  needs ref-load
    "test_qwen2.5_0.5B_opd_vllm.py:8"                 # ✅
    "test_qwen2.5_0.5B_debug_rollout_then_train.py:8" # ✅
    "test_qwen2_5_0_5B_non_colocate_pp.py:4"          # ✅
)

# Group 2: Qwen3-4B (medium, ~3–8 min each)
Q34B_TESTS=(
    "test_qwen3_4B_ppo.py:8"                          # ❌  needs ref-load
    "test_qwen3_4B_ppo_train_critic_only.py:8"        # ❌  needs ref-load
    "test_qwen3_4B_ppo_disaggregate.py:8"             # ❌  needs ref-load
    "test_qwen3_4B_streaming_partial_rollout.py:8"    # ❌  needs ref-load
    "test_qwen3_4B_external_pd.py:6"                  # ✅  has convert_checkpoint in prepare()
)

# Group 3: Utility / GPU misc (fast, ~1–3 min each)
UTIL_TESTS=(
    "test_vllm_rollout.py:4"                          # ❌  3 subtest failures
    "test_external_vllm_engines.py:4"                 # ✅
    "test_empty_colocated_weight_bucket.py:4"         # ✅
    "test_ppo_logprob_entropy_gpu.py:2"               # ✅
    "test_qwen3_5_mtp_bridge_mapping.py:4"            # ✅
    "test_qwen3_linear_attention_cu_seqlens.py:4"     # ✅
    "test_release_train.py:4"                         # ✅
)

# Group 4: Large models (30B+, known hangs)
BIG_TESTS=(
    "test_qwen3_30B_A3B_r3.py:8"                     # ❌  aiter CK MoE device_gemm
    "test_qwen3_0.6B_parallel_check.py:8"             # ✅
    "test_gemma4_12B_gsm8k_short.py:4"               # ❌  gemma4_unified not registered
    "test_mimo_7B_mtp_only_grad.py:8"                 # ❌  MiMoConfig attr error
    "test_moonlight_16B_A3B.py:8"                     # ❌  aiter CK MoE device_gemm
    "test_moonlight_16B_A3B_r3.py:8"                  # ❌  aiter CK MoE device_gemm
)

# Tests that are known to pass (for default run)
PASSING_TESTS=(
    "test_qwen2.5_0.5B_opd_vllm.py:8"
    "test_qwen2.5_0.5B_debug_rollout_then_train.py:8"
    "test_qwen2_5_0_5B_non_colocate_pp.py:4"
    "test_qwen3_4B_external_pd.py:6"
    "test_external_vllm_engines.py:4"
    "test_empty_colocated_weight_bucket.py:4"
    "test_ppo_logprob_entropy_gpu.py:2"
    "test_qwen3_5_mtp_bridge_mapping.py:4"
    "test_qwen3_linear_attention_cu_seqlens.py:4"
    "test_release_train.py:4"
    "test_qwen3_0.6B_parallel_check.py:8"
)

# Buildkite pipeline suites (mirrors gen_rocm_pipeline.py)
SHORT_SUITE=(
    "test_qwen2.5_0.5B_fully_async_short.py:4"
    "test_qwen3.5_0.8B_gsm8k_short.py:4"
    "test_qwen3.5_0.8B_gsm8k_async_short.py:4"
)

MEGATRON_SUITE=(
    "test_full_disk_weight_update.py:4"
    "test_quick_start_glm4_9B.py:8"
    "test_qwen3_30B_A3B.py:8:-e VIME_TEST_USE_DEEPEP=1 -e VIME_TEST_USE_FP8_ROLLOUT=1"
    "test_qwen3_4B_ppo.py:8"
    "test_moonlight_16B_A3B.py:8"
    "test_qwen3_4B_external_pd.py:6"
)

CKPT_SUITE=(
    "test_qwen3_4B_ckpt.py:8:--save-optimizer gpu --load-optimizer gpu"
    "test_qwen3_4B_ckpt.py:8:--save-optimizer cpu --load-optimizer cpu"
    "test_qwen3_4B_ckpt.py:8:--async-save"
)

ALL_24_TESTS=("${Q25_TESTS[@]}" "${Q34B_TESTS[@]}" "${UTIL_TESTS[@]}" "${BIG_TESTS[@]}")

# ── Helpers ───────────────────────────────────────────────────────────────

red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
yellow(){ echo -e "\033[33m$*\033[0m"; }
bold()  { echo -e "\033[1m$*\033[0m"; }

die() { red "ERROR: $*"; exit 1; }

check_prereqs() {
    command -v docker &>/dev/null || die "docker not found"
    docker info &>/dev/null || die "docker daemon not running"

    if [ ! -e /dev/kfd ] || [ ! -e /dev/dri ]; then
        die "/dev/kfd or /dev/dri not found — ROCm kernel driver missing"
    fi

    if ! docker image inspect "${VIME_ROCM_IMAGE}" &>/dev/null; then
        echo "Pulling ${VIME_ROCM_IMAGE}..."
        docker pull "${VIME_ROCM_IMAGE}"
    fi

    mkdir -p "${HF_CACHE}" "${MODELS_DIR}"
}

start_container() {
    echo "Starting container: ${CONTAINER_NAME}"
    docker run -d --name "${CONTAINER_NAME}" \
        --device=/dev/kfd --device=/dev/dri \
        --security-opt seccomp=unconfined --group-add video \
        --ipc=host --shm-size=16g \
        --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
        -e HF_HOME=/root/.cache/huggingface \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${MODELS_DIR}:/root/models" \
        -v "${SCRIPT_DIR}:/root/vime" \
        -w /root/vime \
        "${VIME_ROCM_IMAGE}"

    # Wait for GPU access
    echo "Waiting for GPU access..."
    for i in $(seq 1 30); do
        if docker exec "${CONTAINER_NAME}" python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null; then
            break
        fi
        sleep 2
    done

    # Install vime in editable mode
    docker exec "${CONTAINER_NAME}" pip install -e . --no-deps --break-system-packages -q
    echo "Container ready."
}

stop_container() {
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}

run_single_test_in_container() {
    local test_file="$1"
    docker exec \
        -e VIME_AMD_ROCM=1 \
        -e VIME_TEST_DEVICE=rocm \
        -e VIME_SCRIPT_EXTERNAL_RAY=0 \
        -e VIME_TEST_ENABLE_INFINITE_RUN=false \
        -e VIME_TEST_USE_DEEPEP=0 \
        -e VIME_TEST_USE_FP8_ROLLOUT=0 \
        -e VIME_TEST_ENABLE_EVAL=1 \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        "${CONTAINER_NAME}" \
        /root/vime/run_test_in_container.sh "${test_file}"
}

run_single_test() {
    local test_file="$1"
    local num_gpus="${2:-$NUM_GPUS_DEFAULT}"
    local -a extra_env=()
    local -a extra_args=()

    shift 2 2>/dev/null || true
    while [ $# -gt 0 ]; do
        if [[ "$1" == -e ]]; then
            extra_env+=(-e "$2")
            shift 2
        else
            extra_args+=("$1")
            shift
        fi
    done

    local test_name="${test_file%.py}"
    bold "▶ Running: ${test_name} (${num_gpus} GPU)"

    # Build docker exec command with env and args
    local -a exec_cmd=(
        docker exec
        -e VIME_AMD_ROCM=1
        -e VIME_TEST_DEVICE=rocm
        -e VIME_SCRIPT_EXTERNAL_RAY=0
        -e VIME_TEST_ENABLE_INFINITE_RUN=false
        -e VIME_TEST_USE_DEEPEP=0
        -e VIME_TEST_USE_FP8_ROLLOUT=0
        -e VIME_TEST_ENABLE_EVAL=1
        -e HF_HOME=/root/.cache/huggingface
        -e HF_TOKEN="${HF_TOKEN:-}"
        "${extra_env[@]}"
        "${CONTAINER_NAME}"
        python3 "tests/ci/gpu_lock_exec.py"
            --count "${num_gpus}"
            --target-env-name HIP_VISIBLE_DEVICES
            -- python3 "tests/${test_file}" "${extra_args[@]:-}"
    )

    "${exec_cmd[@]}"
    local rc=$?
    if [ $rc -eq 0 ]; then
        green "  ✅ ${test_name} PASSED"
    else
        red "  ❌ ${test_name} FAILED (exit ${rc})"
    fi
    return $rc
}

run_suite() {
    local suite_name="$1"
    shift
    local tests=("$@")

    bold "═══ Suite: ${suite_name} ($(echo "${tests[@]}" | wc -w) tests) ═══"
    local passed=0
    local failed=0
    local failed_tests=()

    for entry in "${tests[@]}"; do
        IFS=':' read -r test_file num_gpus extra <<< "${entry}"
        local -a extra_args=()
        if [ -n "${extra:-}" ]; then
            read -ra extra_args <<< "${extra}"
        fi

        if run_single_test "${test_file}" "${num_gpus}" "${extra_args[@]:-}"; then
            ((passed++))
        else
            ((failed++))
            failed_tests+=("${test_file}")
        fi
    done

    echo ""
    bold "Suite ${suite_name}: ${passed} passed, ${failed} failed"
    if [ ${#failed_tests[@]} -gt 0 ]; then
        red "Failed tests:"
        for t in "${failed_tests[@]}"; do
            red "  - ${t}"
        done
    fi
}

# ── Model Prep ────────────────────────────────────────────────────────────

prepare_model() {
    local model_name="$1" model_script="$2"
    yellow "Pre-converting model: ${model_name}"

    docker run --rm \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --privileged \
        --security-opt seccomp=unconfined \
        --ipc=host --shm-size=16g \
        --ulimit memlock=-1 \
        -e VIME_AMD_ROCM=1 -e HF_HOME=/root/.cache/huggingface \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${MODELS_DIR}:/root/models" \
        -v "${SCRIPT_DIR}:/root/vime" \
        -w /root/vime \
        --entrypoint bash "${VIME_ROCM_IMAGE}" -c "
            pip install -e . --no-deps --break-system-packages > /dev/null 2>&1
            python3 -c \"
import os; os.environ['VIME_AMD_ROCM'] = '1'
from vime.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint('${model_name}', '${model_script}',
    num_gpus_per_node=1,
    extra_args='--no-gradient-accumulation-fusion --attention-backend flash',
    dir_dst='/root/models')
\"
            cp -r /root/models/${model_name}_torch_dist/* /root/models/${model_name}/ 2>/dev/null || true
        "
    echo "Model ${model_name} prepared in ${MODELS_DIR}/${model_name}"
}

# ── Main ──────────────────────────────────────────────────────────────────

trap stop_container EXIT

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [test_file.py]

Options:
  --suite SUITE    Run a test suite
  --prep-models    Pre-convert all required models (Qwen2.5-0.5B, Qwen3-4B, Qwen3-8B)
  --help           This message

Suites:
  passing        11 tests known to pass (default)
  short          3-test Buildkite smoke suite
  megatron       6-test megatron suite
  ckpt           3-test checkpoint suite
  q25            7-test Qwen2.5-0.5B group
  q34b           5-test Qwen3-4B group
  util           7-test utility group
  big            6-test large-model group
  all            24 tests (full run)

Env vars:
  VIME_ROCM_IMAGE  Docker image (default: vllm/vime-rocm:latest)
  HF_HOME          HuggingFace cache dir (default: ~/.cache/huggingface)
  MODELS_DIR       Converted Megatron checkpoints (default: ~/models)
  HF_TOKEN         HuggingFace token (avoids rate limits)
EOF
}

SUITE=""
SINGLE_TEST=""
PREP_MODELS=false

while [ $# -gt 0 ]; do
    case "$1" in
        --suite)       SUITE="$2"; shift 2 ;;
        --prep-models) PREP_MODELS=true; shift ;;
        --help|-h)     usage; exit 0 ;;
        *.py)          SINGLE_TEST="$1"; shift ;;
        *)             die "Unknown argument: $1. Try --help." ;;
    esac
done

check_prereqs

if [ "$PREP_MODELS" = true ]; then
    prepare_model "Qwen2.5-0.5B-Instruct" "qwen2.5-0.5B"
    prepare_model "Qwen3-4B" "qwen3-4B"
    prepare_model "Qwen3-8B" "qwen3-8B"
    green "All models prepared. You can now run tests."
    exit 0
fi

if [ -n "${SINGLE_TEST}" ]; then
    start_container
    run_single_test_in_container "${SINGLE_TEST}"
elif [ -n "${SUITE}" ]; then
    start_container
    case "${SUITE}" in
        passing)   run_suite "passing" "${PASSING_TESTS[@]}" ;;
        short)     run_suite "short" "${SHORT_SUITE[@]}" ;;
        megatron)  run_suite "megatron" "${MEGATRON_SUITE[@]}" ;;
        ckpt)      run_suite "ckpt" "${CKPT_SUITE[@]}" ;;
        q25)       run_suite "q25" "${Q25_TESTS[@]}" ;;
        q34b)      run_suite "q34b" "${Q34B_TESTS[@]}" ;;
        util)      run_suite "util" "${UTIL_TESTS[@]}" ;;
        big)       run_suite "big" "${BIG_TESTS[@]}" ;;
        all)       run_suite "all" "${ALL_24_TESTS[@]}" ;;
        *)         die "Unknown suite: ${SUITE}. Try --help." ;;
    esac
else
    # Default: run passing tests + show how to run more
    start_container
    run_suite "passing" "${PASSING_TESTS[@]}"
    echo ""
    yellow "💡 This ran 11 known-passing tests."
    yellow "   To run all 24:  $0 --suite all"
    yellow "   To run a suite:  $0 --suite megatron"
    yellow "   To prep models:  $0 --prep-models"
    yellow "   Full guide: https://app.notion.com/p/VIME-ROCm-GPU-CI-Reproducibility-Guide-3a074178c99781de919ce065cd32919d"
fi
