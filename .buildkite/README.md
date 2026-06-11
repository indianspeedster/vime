# vime CI on Buildkite

Buildkite port of the **always-on (CPU) jobs** from
`.github/workflows/pr-test.yml.j2`. The GitHub Actions workflow keeps running
in parallel and stays authoritative until Buildkite has proven itself; the
label-gated GPU suites are not migrated yet.

The always-on steps live in the static [`pipeline.yml`](./pipeline.yml) and
run on every build (PR and push to `main`):

| Step | Mirrors GHA job | Queue (machine) |
|---|---|---|
| `pre-commit` | `pre-commit` gate | `small_cpu_queue_premerge` (r6in.large) |
| `plugin-contracts` | `e2e-test-plugin-contracts` (19 files) | `medium_cpu_queue_premerge` (r6in.4xlarge) |
| `agent-adapter` | `agent-adapter-test` (3 files) | `small_cpu_queue_premerge` |
| `unit` | `e2e-test-unit` (`pytest tests/unit tests/utils`) | `medium_cpu_queue_premerge` |

The three test steps `depends_on` the pre-commit gate, matching the GHA
`needs: pre-commit`. Each suite runs its files sequentially inside one step
because these queues boot a fresh EC2 instance per job — a per-file matrix
would be mostly boot + pip-install time. The `unit` step pulls
`inferactinc/public:vime-latest` on every build (no local image cache on
ephemeral instances); if pull time becomes a problem, mirror the image to ECR
(the premerge queues already have read-only ECR access).

## Creating the pipeline (one-time, Buildkite UI)

Org `vllm`, cluster **CI** (the premerge queues live there).

1. New pipeline: name `vime-ci`, repository
   `https://github.com/vllm-project/vime.git`.
2. Leave the pipeline's Steps field as the default upload step — it reads the
   committed `.buildkite/pipeline.yml`:

   ```yaml
   steps:
     - command: buildkite-agent pipeline upload
       agents:
         queue: small_cpu_queue_premerge
   ```

3. GitHub settings on the pipeline:
   - Trigger builds after pushing code; branch filter: `main`.
   - Build pull requests (same-repository PRs only); skip builds for existing
     commits.
   - Update commit statuses.
   The Buildkite GitHub app must have access to `vllm-project/vime`.
4. Pipeline settings: enable **Skip Intermediate Builds** and
   **Cancel Intermediate Builds** (replaces the GHA concurrency group).

No secrets are required for these steps (WANDB etc. is GPU-suite only).

## GPU suites (manual gate instead of PR labels)

GitHub PR labels can't trigger Buildkite jobs, so the `run-ci-*` label-gated
GPU suites are behind a **block step** (`:rocket: Run GPU test suites?`):
click it in the Buildkite UI, multi-select the suites (`short`,
`vllm-config`, `megatron`, `precision`, `ckpt`), and the follow-up step
generates one job per test via [`gpu_suites.py`](./gpu_suites.py) — the same
`gpu_lock_exec.py` + `docker run` invocations as the GHA jobs, including the
per-test `VIME_TEST_USE_DEEPEP` / `USE_FP8_ROLLOUT` / `ENABLE_EVAL` combos.

The block uses `blocked_state: passed`, so a build whose CPU steps are green
reports a passing commit status even if nobody unblocks the GPU gate.

GPU jobs target the agent queue **`vime-gpu`**. Prerequisites on each vime
self-hosted GPU host:

- a `buildkite-agent` registered in the **CI** cluster with
  `tags="queue=vime-gpu"`,
- `WANDB_API_KEY` exported in the agent's environment (an agent `environment`
  hook works),
- the usual `/mnt/nvme0n1/vime_ci` model/dataset mounts.

## Keeping it in sync

The test lists mirror `.github/workflows/pr-test.yml.j2` (always-on jobs in
`pipeline.yml`, label-gated jobs in `gpu_suites.py`). Until the GHA jobs are
retired, a test added/removed there should be mirrored here.
