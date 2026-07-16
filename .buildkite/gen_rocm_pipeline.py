#!/usr/bin/env python3
"""Generate .buildkite/pipeline-rocm.yaml (declarative) from the suite table.

Mirrors the CUDA .buildkite/gpu_suites.py SUITES, but emits a static, readable
Buildkite YAML (per-suite manual block gate + docker steps that arbitrate GPUs
via gpu_lock_exec on HIP_VISIBLE_DEVICES). The shared docker run is a YAML
anchor (&rocm_gpu_test) defined once and aliased everywhere.

This generator is a dev tool — only the emitted YAML is committed.
"""

# (test_file, num_gpus, extra_args, env-overrides, soft_fail)
SUITES = {
    "short": [
        ("test_qwen2.5_0.5B_fully_async_short.py", 4, "", {}, False),
        ("test_qwen3.5_0.8B_gsm8k_short.py", 4, "", {}, True),
        ("test_qwen3.5_0.8B_gsm8k_async_short.py", 4, "", {}, True),
    ],
    "vllm-config": [
        ("test_qwen2.5_0.5B_vllm_config.py", 8, "", {}, True),
        ("test_qwen2.5_0.5B_vllm_config_distributed.py", 8, "", {}, True),
        ("test_vllm_config_mixed_offload.py", 8, "", {}, True),
        ("test_vllm_config_mixed_offload_ft.py", 8, "", {}, True),
    ],
    "megatron": [
        ("test_full_disk_weight_update.py", 4, "", {}, True),
        ("test_quick_start_glm4_9B.py", 8, "", {}, True),
        ("test_glm4.7_30B_A3B_pd_mooncake.py", 8, "", {}, True),
        ("test_qwen3_30B_A3B.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1"}, True),
        ("test_qwen3.6_35B_A3B_pd_mooncake.py", 8, "", {"USE_DEEPEP": "1"}, True),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1", "ENABLE_EVAL": "0"}, True),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}, True),
        ("test_qwen3_4B_ppo.py", 8, "", {}, True),
        ("test_qwen3_4B_ppo_disaggregate.py", 8, "", {}, True),
        ("test_qwen3_4B_ppo_train_critic_only.py", 8, "", {}, True),
        ("test_ppo_logprob_entropy_gpu.py", 2, "", {}, True),
        ("test_release_train.py", 4, "", {}, True),
        ("test_qwen3_4B_streaming_partial_rollout.py", 8, "", {}, True),
        ("test_moonlight_16B_A3B.py", 8, "", {}, True),
        ("test_moonlight_16B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}, True),
        ("test_mimo_7B_mtp_only_grad.py", 8, "", {}, True),
        ("test_qwen2.5_0.5B_debug_rollout_then_train.py", 8, "", {}, True),
        ("test_qwen2.5_0.5B_opd_vllm.py", 8, "", {}, True),
        ("test_qwen3_4B_external_pd.py", 6, "", {}, True),
        ("test_qwen2.5_0.5B_fanout_short.py", 4, "", {}, True),
    ],
    "vime-customized": [
        ("test_qwen2_5_0_5B_non_colocate_pp.py", 4, "", {}, True),
    ],
    "precision": [
        ("test_qwen3_0.6B_parallel_check.py", 8, "", {}, True),
    ],
    "ckpt": [
        ("test_qwen3_4B_ckpt.py", 8, "--save-optimizer gpu --load-optimizer gpu", {}, True),
        ("test_qwen3_4B_ckpt.py", 8, "--save-optimizer gpu --load-optimizer cpu", {}, True),
        ("test_qwen3_4B_ckpt.py", 8, "--save-optimizer cpu --load-optimizer cpu", {}, True),
        ("test_qwen3_4B_ckpt.py", 8, "--save-optimizer cpu --load-optimizer gpu", {}, True),
        ("test_qwen3_4B_ckpt.py", 8, "--async-save", {}, True),
    ],
}

# Env keys the tests read as VIME_TEST_<KEY>. Values default in the tests, so we
# only emit a step-level override when the suite table sets one.
FLAG_KEYS = {"USE_DEEPEP", "USE_FP8_ROLLOUT", "ENABLE_EVAL"}

HEADER = """# Buildkite CI for vime — AMD ROCm GPU suites.
#
# GENERATED FILE — edit .buildkite/gen_rocm_pipeline.py and regenerate, or hand-edit
# with care. Declarative ROCm analogue of the CUDA .buildkite/gpu_suites.py:
# one manual block gate per suite (so an expensive suite runs only when you
# unblock it), then one docker step per test on the self-hosted AMD agent queue.
# Each step runs the prebuilt ROCm image, passes the ROCm devices through, and
# arbitrates GPUs with tests/ci/gpu_lock_exec.py on HIP_VISIBLE_DEVICES. Tests
# take the U.is_rocm() path (HF->Megatron conversion, no bridge).
#
# One-time setup (see .buildkite/README.md): a Buildkite agent on the gfx950
# (MI350X) node tagged  queue=amd_gfx950  with docker + the ROCm devices.
#
# soft_fail mirrors the CUDA SOFT_FAIL set / vllm-omni's grade: only the
# validated short/fully_async test blocks; every other suite is newly ported and
# not yet validated on ROCm, so it runs visible (orange) without failing the
# build. Drop soft_fail from a step once it passes on real hardware.

env:
  # Prebuilt ROCm image (docker/Dockerfile.rocm). Override per-agent if needed.
  VIME_ROCM_IMAGE: "vllm/vime-rocm:latest"
  VIME_TEST_ENABLE_INFINITE_RUN: "false"
  # Defaults for the flags the suite table can override per-step.
  VIME_TEST_USE_DEEPEP: "0"
  VIME_TEST_USE_FP8_ROLLOUT: "0"
  VIME_TEST_ENABLE_EVAL: "1"
  EXTRA_ARGS: ""

steps:
"""

# The shared docker run. $$VAR is escaped so the agent expands it at run time
# (from step env / checkout) instead of Buildkite interpolating it at upload.
# The inner bash -lc is single-quoted, so $NUM_GPUS/$TEST_FILE/$EXTRA_ARGS are
# expanded by the container shell from the forwarded (-e) env. EXTRA_ARGS is
# intentionally unquoted inside the container so multi-flag values word-split.
COMMAND = """          docker run --rm \\
            --device=/dev/kfd --device=/dev/dri --group-add video --privileged \\
            --security-opt seccomp=unconfined --ipc=host --shm-size=16g \\
            --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \\
            -e VIME_AMD_ROCM=1 -e VIME_TEST_DEVICE=rocm -e VIME_SCRIPT_EXTERNAL_RAY=0 \\
            -e HF_HOME=/root/.cache/huggingface \\
            -e GITHUB_COMMIT_NAME="$$BUILDKITE_COMMIT" \\
            -e TEST_FILE -e NUM_GPUS -e EXTRA_ARGS -e VIME_TEST_ENABLE_INFINITE_RUN \\
            -e VIME_TEST_USE_DEEPEP -e VIME_TEST_USE_FP8_ROLLOUT -e VIME_TEST_ENABLE_EVAL \\
            -v "/root/.cache/huggingface:/root/.cache/huggingface" \\
            -v "$$PWD:/root/vime" -w /root/vime \\
            --entrypoint bash "$$VIME_ROCM_IMAGE" -lc '
              set -euo pipefail
              pip install -e . --no-deps --break-system-packages
              python tests/ci/gpu_lock_exec.py \\
                --count "$$NUM_GPUS" --target-env-name HIP_VISIBLE_DEVICES \\
                -- python "tests/$$TEST_FILE" $${EXTRA_ARGS:-}
            \'"""

SUITE_EMOJI = {
    "short": ":fire:",
    "vllm-config": ":gear:",
    "megatron": ":brain:",
    "vime-customized": ":wrench:",
    "precision": ":triangular_ruler:",
    "ckpt": ":floppy_disk:",
}


def slug(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append(ch if ch.isalnum() else "-")
    r = "".join(out)
    while "--" in r:
        r = r.replace("--", "-")
    return r.strip("-")


def main() -> None:
    lines = [HEADER.rstrip("\n")]
    first_command = True
    seen_keys = set()

    for suite, entries in SUITES.items():
        gate_key = f"gate-{slug(suite)}"
        lines.append("")
        lines.append(f'  - block: "{SUITE_EMOJI[suite]} Run {suite} suite?"')
        lines.append(f"    key: {gate_key}")
        lines.append("    blocked_state: passed")
        lines.append("")
        lines.append(f'  - group: "{SUITE_EMOJI[suite]} {suite}"')
        lines.append(f"    depends_on: {gate_key}")
        lines.append("    steps:")

        for test_file, num_gpus, extra_args, env, soft in entries:
            flag_note = ",".join(f"{k.lower()}={v}" for k, v in env.items() if k in FLAG_KEYS)
            arg_note = f" {extra_args}" if extra_args else ""
            note = ""
            if extra_args:
                note += arg_note
            if flag_note:
                note += f" ({flag_note})"
            emoji = ":warning:" if soft else ":fire:"
            label = f"{emoji} {suite}: {test_file}{note}"

            # unique step key
            base = f"{slug(suite)}-{slug(test_file.replace('.py', ''))}"
            key = base
            n = 2
            while key in seen_keys:
                key = f"{base}-{n}"
                n += 1
            seen_keys.add(key)

            lines.append(f'      - label: "{label}"')
            lines.append(f"        key: {key}")
            lines.append("        agents:")
            lines.append("          queue: amd_gfx950")
            lines.append("        timeout_in_minutes: 360")
            if soft:
                lines.append("        soft_fail: true")
            lines.append("        retry:")
            lines.append("          automatic:")
            lines.append("            - exit_status: -1  # agent lost")
            lines.append("              limit: 2")
            lines.append("        env:")
            lines.append(f'          TEST_FILE: "{test_file}"')
            lines.append(f'          NUM_GPUS: "{num_gpus}"')
            if extra_args:
                lines.append(f'          EXTRA_ARGS: "{extra_args}"')
            for k in ("USE_DEEPEP", "USE_FP8_ROLLOUT", "ENABLE_EVAL"):
                if k in env:
                    lines.append(f'          VIME_TEST_{k}: "{env[k]}"')
            if first_command:
                lines.append("        command: &rocm_gpu_test |")
                lines.append(COMMAND)
                first_command = False
            else:
                lines.append("        command: *rocm_gpu_test")

    print("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
