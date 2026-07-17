#!/usr/bin/env bash
# =============================================================================
# run_test.sh — Reproducible VIME ROCm test runner (no Buildkite required)
#
# Usage:
#   ./run_test.sh                          # run all GPU suite tests
#   ./run_test.sh test_qwen2.5_0.5B_fully_async_short.py  # single test
#   ./run_test.sh --suite short            # run the "short" suite
#
# Requirements:
#   - AMD GPU with ROCm (MI300X / MI350X / gfx942+)
#   - Docker with GPU access (--device=/dev/kfd --device=/dev/dri)
#   - ~200 GB disk for models + HF cache
#
# Quick start:
#   1. docker pull vllm/vime-rocm:latest
#   2. export HF_HOME=/path/to/huggingface_cache  # default: ~/.cache/huggingface
#   3. ./run_test.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ── Config ────────────────────────────────────────────────────────────────
VIME_ROCM_IMAGE="${VIME_ROCM_IMAGE:-vllm/vime-rocm:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CONTAINER_NAME="vime-test-$(date +%Y%m%d-%H%M%S)"
NUM_GPUS_DEFAULT="${NUM_GPUS:-1}"  # per-test default; suites override

# ── Suite definitions (mirrors .buildkite/gen_rocm_pipeline.py SUITES) ────
# Format: "test_file:num_gpus:extra_args"
SHORT_TESTS=(
    "test_qwen2.5_0.5B_fully_async_short.py:4"
    "test_qwen3.5_0.8B_gsm8k_short.py:4"
    "test_qwen3.5_0.8B_gsm8k_async_short.py:4"
)

MEGATRON_TESTS=(
    "test_full_disk_weight_update.py:4"
    "test_quick_start_glm4_9B.py:8"
    "test_qwen3_30B_A3B.py:8:-e VIME_TEST_USE_DEEPEP=1 -e VIME_TEST_USE_FP8_ROLLOUT=1"
    "test_qwen3_4B_ppo.py:8"
    "test_moonlight_16B_A3B.py:8"
    "test_qwen3_4B_external_pd.py:6"
)

CKPT_TESTS=(
    "test_qwen3_4B_ckpt.py:8:--save-optimizer gpu --load-optimizer gpu"
    "test_qwen3_4B_ckpt.py:8:--save-optimizer cpu --load-optimizer cpu"
    "test_qwen3_4B_ckpt.py:8:--async-save"
)

ALL_TESTS=("${SHORT_TESTS[@]}" "${MEGATRON_TESTS[@]}" "${CKPT_TESTS[@]}")

# ── Helpers ───────────────────────────────────────────────────────────────

red()  { echo -e "\033[31m$*\033[0m"; }
green(){ echo -e "\033[32m$*\033[0m"; }
bold() { echo -e "\033[1m$*\033[0m"; }

die() { red "ERROR: $*"; exit 1; }

check_prereqs() {
    command -v docker &>/dev/null || die "docker not found"
    docker info &>/dev/null || die "docker daemon not running"

    # Check ROCm devices exist on host
    if [ ! -e /dev/kfd ] || [ ! -e /dev/dri ]; then
        die "/dev/kfd or /dev/dri not found — ROCm kernel driver missing"
    fi

    # Pull image if needed
    if ! docker image inspect "${VIME_ROCM_IMAGE}" &>/dev/null; then
        echo "Pulling ${VIME_ROCM_IMAGE}..."
        docker pull "${VIME_ROCM_IMAGE}"
    fi

    # Ensure HF cache directory exists
    mkdir -p "${HF_CACHE}"
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

run_single_test() {
    local test_file="$1"
    local num_gpus="${2:-$NUM_GPUS_DEFAULT}"
    local extra_env=()
    local extra_args=()

    # Parse extra args: either env vars (-e KEY=VAL) or CLI args
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

    local test_name="$(basename "${test_file}" .py)"
    bold "▶ Running: ${test_name} (${num_gpus} GPU)"

    docker exec \
        -e VIME_TEST_DEVICE=rocm \
        -e VIME_SCRIPT_EXTERNAL_RAY=0 \
        "${extra_env[@]}" \
        "${CONTAINER_NAME}" \
        python3 "tests/ci/gpu_lock_exec.py" \
            --count "${num_gpus}" \
            --target-env-name HIP_VISIBLE_DEVICES \
            -- python3 "tests/${test_file}" ${extra_args[@]:-}

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

    bold "═══ Suite: ${suite_name} ($(echo "${tests[@]}" | wc -w | xargs) tests) ═══"
    local passed=0
    local failed=0
    local failed_tests=()

    for entry in "${tests[@]}"; do
        IFS=':' read -r test_file num_gpus extra <<< "${entry}"
        local extra_env=()
        local extra_args=()
        if [ -n "${extra:-}" ]; then
            # Could be env vars (-e KEY=VAL) or CLI args
            read -ra parts <<< "${extra}"
            for part in "${parts[@]}"; do
                if [[ "$part" == -e ]]; then
                    # handled in loop below
                    :
                fi
            done
            extra_args="${extra}"
        fi

        if run_single_test "${test_file}" "${num_gpus}" ${extra_args:-}; then
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

# ── Main ──────────────────────────────────────────────────────────────────

trap stop_container EXIT

# Parse arguments
SUITE=""
SINGLE_TEST=""

while [ $# -gt 0 ]; do
    case "$1" in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--suite short|megatron|ckpt|all] [test_file.py]"
            echo ""
            echo "Suites:"
            echo "  short     3 tests  (~10 min)"
            echo "  megatron  16 tests (~4 hours)"
            echo "  ckpt      5 tests  (~1 hour)"
            echo "  all       all tests (~5 hours)"
            echo ""
            echo "Env vars:"
            echo "  VIME_ROCM_IMAGE  Docker image (default: vllm/vime-rocm:latest)"
            echo "  HF_HOME          HuggingFace cache dir (default: ~/.cache/huggingface)"
            exit 0
            ;;
        *.py)
            SINGLE_TEST="$1"
            shift
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

check_prereqs

if [ -n "${SINGLE_TEST}" ]; then
    start_container
    run_single_test "${SINGLE_TEST}" "${NUM_GPUS_DEFAULT}"
elif [ -n "${SUITE}" ]; then
    start_container
    case "${SUITE}" in
        short)    run_suite "short" "${SHORT_TESTS[@]}" ;;
        megatron) run_suite "megatron" "${MEGATRON_TESTS[@]}" ;;
        ckpt)     run_suite "ckpt" "${CKPT_TESTS[@]}" ;;
        all)      run_suite "all" "${ALL_TESTS[@]}" ;;
        *)        die "Unknown suite: ${SUITE}. Try: short, megatron, ckpt, all" ;;
    esac
else
    # Default: run short suite
    start_container
    run_suite "short" "${SHORT_TESTS[@]}"
    echo ""
    bold "To run more tests: $0 --suite megatron  or  $0 --suite all"
fi
