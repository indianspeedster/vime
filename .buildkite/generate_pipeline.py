#!/usr/bin/env python3
"""Generate the Buildkite pipeline for vime's always-on (CPU) CI jobs.

Buildkite analogue of the always-on jobs in .github/workflows/pr-test.yml.j2,
which stays authoritative while GitHub Actions and Buildkite run in parallel:

  pre-commit        lint gate; every build
  plugin-contracts  CPU test files; every build (including push to main)
  agent-adapter     CPU test files; PR / manual branch builds only
  unit              pytest tests/unit tests/utils inside the CI image;
                    PR / manual branch builds only

GPU suites (run-ci-* labels) are not migrated yet.

Design notes:
  * Buildkite step `if:` conditions can't see GitHub events or PR labels, so
    all event logic lives here: the generator inspects BUILDKITE_BRANCH /
    BUILDKITE_PULL_REQUEST and only emits the steps that should run.
  * The vLLM elastic-stack CPU queues terminate the EC2 instance after every
    job, so each suite is ONE step running its files sequentially instead of
    a per-file matrix (a fresh instance + pip install per file is all boot
    time, no test time).
  * Steps run inside containers (python:3.10 to match the GHA jobs, or the
    vime CI image for the unit suite) because the host AMI's python is not
    pinned.

stdlib only — the bootstrap step runs this with the host python3.
"""

import json
import os
import shlex

CPU_QUEUE_SMALL = "small_cpu_queue_premerge"  # r6in.large, 2 vCPU / 16 GB
CPU_QUEUE_MEDIUM = "medium_cpu_queue_premerge"  # r6in.4xlarge, 16 vCPU / 128 GB

PYTHON_IMAGE = "python:3.10"
CI_IMAGE = "inferactinc/public:vime-latest"
REPO_MOUNT = "/workspace"

# Test lists mirror the always-on jobs in .github/workflows/pr-test.yml.j2 —
# keep in sync until the GHA always-on jobs are retired and this file becomes
# the single source of truth.
PLUGIN_CONTRACT_TESTS = [
    "test_megatron_argument_validation.py",
    "test_value_temperature.py",
    "test_rollout_validation.py",
    "plugin_contracts/test_plugin_rollout_contracts.py",
    "plugin_contracts/test_plugin_runtime_hook_contracts.py",
    "plugin_contracts/test_plugin_path_loading_contracts.py",
    "plugin_contracts/test_plugin_generate_contracts.py",
    "test_rm_deepscaler.py",
    "test_rm_f1.py",
    "test_rm_gpqa.py",
    "test_rm_math.py",
    "test_rm_math_dapo.py",
    "test_dp_schedule.py",
    "test_cp_utils.py",
    "test_metric_report.py",
    "test_metric_report_dist.py",
    "test_loss_cp_invariance.py",
    "test_sample.py",
    "utils/test_hf_checkpoint_saver.py",
]

AGENT_ADAPTER_TESTS = [
    "test_agent_trajectory.py",
    "test_agent_adapters.py",
    "test_agent_sdk_adapters.py",
]

CPU_TEST_DEPS = "pytest numpy packaging pyyaml omegaconf tqdm httpx pybase64 " "pylatexenc sympy aiohttp pillow"
AGENT_ADAPTER_EXTRA_DEPS = "openai openai-agents anthropic"


def docker_command(image: str, script: str, extra_args: str = "") -> str:
    """`docker run` wrapper matching the GHA jobs' raw-docker style."""
    return (
        "docker run --rm --ipc=host --shm-size=4g "
        f"{extra_args}"
        f'-v "$PWD:{REPO_MOUNT}" -w {REPO_MOUNT} '
        f"{image} bash -c {shlex.quote(script)}"
    )


def test_loop(test_files: list) -> str:
    """Run every file, then fail with the full list of broken ones.

    `--- ` / `+++ ` are Buildkite log group markers (+++ auto-expands).
    A `cmd || handler` compound does not trip `set -e`, so the loop always
    finishes even with -e active for the setup above it.
    """
    lines = ["failed=()"]
    for test in test_files:
        lines += [
            f'echo "--- {test}"',
            f'python "tests/{test}" || {{ echo "^^^ +++"; failed+=("{test}"); }}',
        ]
    lines += [
        'if [ "${#failed[@]}" -gt 0 ]; then',
        '  echo "+++ Failed test files:"',
        '  printf "  %s\\n" "${failed[@]}"',
        "  exit 1",
        "fi",
    ]
    return "\n".join(lines)


def step(label: str, key: str, queue: str, command: str, timeout: int, gated: bool = True) -> dict:
    s = {
        "label": label,
        "key": key,
        "command": command,
        "agents": {"queue": queue},
        "timeout_in_minutes": timeout,
        # Fresh instance per job: retry once if the agent is lost mid-boot.
        "retry": {"automatic": [{"exit_status": -1, "limit": 2}]},
    }
    if gated:
        s["depends_on"] = "pre-commit"
    return s


def pre_commit_step() -> dict:
    script = "\n".join(
        [
            "set -euo pipefail",
            # The checkout is owned by the host's buildkite-agent user; the
            # container runs as root, so git refuses to touch it without this.
            f"git config --global --add safe.directory {REPO_MOUNT}",
            "pip install -q pre-commit",
            "pre-commit run --all-files --show-diff-on-failure --color=always",
        ]
    )
    return step(
        ":lint-roller: pre-commit",
        "pre-commit",
        CPU_QUEUE_SMALL,
        docker_command(PYTHON_IMAGE, script),
        timeout=15,
        gated=False,
    )


def cpu_suite_step(label: str, key: str, queue: str, tests: list, extra_deps: str = "") -> dict:
    setup = [
        "set -euo pipefail",
        'echo "--- pip install"',
        "pip install -q torch --index-url https://download.pytorch.org/whl/cpu",
        f"pip install -q {CPU_TEST_DEPS}",
    ]
    if extra_deps:
        setup.append(f"pip install -q {extra_deps}")
    setup.append("pip install -q -e . --no-deps")
    script = "\n".join(setup) + "\n" + test_loop(tests)
    return step(label, key, queue, docker_command(PYTHON_IMAGE, script), timeout=30)


def unit_step() -> dict:
    script = "\n".join(
        [
            "set -euo pipefail",
            'echo "--- pip install"',
            "pip install -q -e . --no-deps --break-system-packages",
            "pip install -q pytest --break-system-packages",
            'echo "+++ pytest tests/unit tests/utils"',
            "python -m pytest tests/unit tests/utils",
        ]
    )
    return step(
        ":pytest: unit & utils tests (in-image)",
        "unit",
        CPU_QUEUE_MEDIUM,
        docker_command(CI_IMAGE, script, extra_args="--network host "),
        timeout=45,
    )


def main() -> None:
    branch = os.environ.get("BUILDKITE_BRANCH", "")
    is_pr = os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false"
    # Mirror the GHA per-job `if:` matrix: push-to-main builds run only the
    # lint gate + plugin contracts (the cheap PR-pair regression check); PR
    # and manually triggered branch builds run everything.
    main_push = branch == "main" and not is_pr

    steps = [
        pre_commit_step(),
        cpu_suite_step(
            ":python: plugin contracts & CPU tests",
            "plugin-contracts",
            CPU_QUEUE_MEDIUM,
            PLUGIN_CONTRACT_TESTS,
        ),
    ]
    if not main_push:
        steps.append(
            cpu_suite_step(
                ":robot_face: agent adapter tests",
                "agent-adapter",
                CPU_QUEUE_SMALL,
                AGENT_ADAPTER_TESTS,
                extra_deps=AGENT_ADAPTER_EXTRA_DEPS,
            )
        )
        steps.append(unit_step())

    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
