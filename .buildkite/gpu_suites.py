#!/usr/bin/env python3
"""Emit Buildkite steps for the GPU suites selected at the gpu-gate block step.

Piped into `buildkite-agent pipeline upload` by the gpu-suites-upload step in
pipeline.yml. The suites and their env-var combinations mirror the label-gated
GPU jobs in .github/workflows/pr-test.yml.j2 (run-ci-short / vllm-config /
megatron / precision / ckpt); keep them in sync until the GHA jobs are retired.

The selection is read from the block step's multi-select field (newline-
separated values in the `gpu-suites` build meta-data key). For local testing,
set GPU_SUITES=short,ckpt instead of having a buildkite-agent on PATH.

stdlib only — runs with the agent host's python3.
"""

import json
import os
import shlex
import subprocess

GPU_QUEUE = "vime-gpu"  # self-hosted vime GPU hosts registered with this tag
CI_IMAGE = "inferactinc/public:vime-latest"

# (test_file, num_gpus, extra_args, env overrides)
SUITES = {
    "short": [
        ("test_qwen3.5_0.8B_gsm8k_async_short.py", 4, "", {}),
        ("test_qwen3.5_0.8B_gsm8k_short.py", 4, "", {}),
        ("test_qwen2.5_0.5B_ppo_critic_only_short.py", 4, "", {}),
        ("test_qwen2.5_0.5B_fully_async_short.py", 4, "", {}),
    ],
    "vllm-config": [
        ("test_qwen2.5_0.5B_vllm_config.py", 8, "", {}),
        ("test_qwen2.5_0.5B_vllm_config_distributed.py", 8, "", {}),
        ("test_vllm_config_mixed_offload.py", 8, "", {}),
        ("test_vllm_config_mixed_offload_ft.py", 8, "", {}),
    ],
    "megatron": [
        ("test_quick_start_glm4_9B.py", 8, "", {}),
        ("test_glm4.7_30B_A3B_pd_mooncake.py", 8, "", {}),
        ("test_qwen3_30B_A3B.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1"}),
        ("test_qwen3.6_35B_A3B_pd_mooncake.py", 8, "", {"USE_DEEPEP": "1"}),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1", "ENABLE_EVAL": "0"}),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}),
        ("test_qwen3_4B_ppo.py", 8, "", {}),
        ("test_qwen3_4B_ppo_disaggregate.py", 8, "", {}),
        ("test_qwen3_4B_ppo_train_critic_only.py", 8, "", {}),
        ("test_qwen3_4B_streaming_partial_rollout.py", 8, "", {}),
        ("test_moonlight_16B_A3B.py", 8, "", {}),
        ("test_moonlight_16B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}),
        ("test_qwen2.5_0.5B_debug_rollout_then_train.py", 8, "", {}),
        ("test_qwen2.5_0.5B_opd_vllm.py", 8, "", {}),
    ],
    "precision": [
        ("test_qwen3_0.6B_parallel_check.py", 8, "", {}),
    ],
    "ckpt": [
        ("test_qwen3_4B_ckpt.py", 8, "", {}),
        ("test_qwen3_4B_ckpt.py", 8, "--async-save", {}),
    ],
}


def selected_suites() -> list:
    raw = os.environ.get("GPU_SUITES")
    if raw is None:
        raw = subprocess.run(
            ["buildkite-agent", "meta-data", "get", "gpu-suites"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    # multi-select meta-data is newline-separated; accept commas too
    values = [v.strip() for v in raw.replace(",", "\n").splitlines()]
    unknown = [v for v in values if v and v not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
    return [s for s in SUITES if s in values]


def gpu_step(suite: str, test_file: str, num_gpus: int, extra_args: str, env: dict) -> dict:
    test_env = {
        "VIME_TEST_ENABLE_INFINITE_RUN": "false",
        "VIME_TEST_USE_DEEPEP": env.get("USE_DEEPEP", "0"),
        "VIME_TEST_USE_FP8_ROLLOUT": env.get("USE_FP8_ROLLOUT", "0"),
        "VIME_TEST_ENABLE_EVAL": env.get("ENABLE_EVAL", "1"),
    }
    inner = "\n".join(
        [
            "set -euo pipefail",
            "pip install -e . --no-deps --break-system-packages",
            f"python tests/ci/gpu_lock_exec.py --count {num_gpus} -- "
            f"python tests/{test_file}{' ' + extra_args if extra_args else ''}",
        ]
    )
    # GITHUB_COMMIT_NAME mirrors GHA: <sha>_<pr-number|non-pr>. WANDB_API_KEY
    # comes from the self-hosted agent's environment.
    command = "\n".join(
        [
            'PR="${BUILDKITE_PULL_REQUEST:-false}"',
            '[ "$PR" = "false" ] && PR="non-pr"',
            'export GITHUB_COMMIT_NAME="${BUILDKITE_COMMIT}_${PR}"',
            "docker run --rm \\",
            "  --privileged --cap-add SYS_NICE --security-opt seccomp=unconfined \\",
            "  --network host --gpus all --ipc=host --shm-size=16g \\",
            "  --ulimit memlock=-1 --ulimit stack=67108864 --memory=0 --memory-swap=0 \\",
            "  -e GITHUB_COMMIT_NAME -e WANDB_API_KEY \\",
        ]
        + [f"  -e {k}={v} \\" for k, v in test_env.items()]
        + [
            '  -v "$PWD:/workspace" -w /workspace \\',
            "  -v /mnt/nvme0n1/vime_ci:/data/vime_ci \\",
            "  -v /mnt/nvme0n1/vime_ci/models:/root/models \\",
            "  -v /mnt/nvme0n1/vime_ci/datasets:/root/datasets \\",
            f"  {CI_IMAGE} bash -lc {shlex.quote(inner)}",
        ]
    )
    label = f":fire: {suite}: {test_file}{' ' + extra_args if extra_args else ''}"
    flag_note = ",".join(f"{k.lower()}={v}" for k, v in env.items())
    if flag_note:
        label += f" ({flag_note})"
    return {
        "label": label,
        "command": command,
        "agents": {"queue": GPU_QUEUE},
        "timeout_in_minutes": 360,
        "retry": {"automatic": [{"exit_status": -1, "limit": 2}]},
    }


def main() -> None:
    steps = [gpu_step(suite, *entry) for suite in selected_suites() for entry in SUITES[suite]]
    if not steps:
        raise SystemExit("no GPU suites selected")
    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
