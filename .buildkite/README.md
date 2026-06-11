# vime CI on Buildkite

Buildkite port of the **always-on (CPU) jobs** from
`.github/workflows/pr-test.yml.j2`. The GitHub Actions workflow keeps running
in parallel and stays authoritative until Buildkite has proven itself; the
label-gated GPU suites are not migrated yet.

| Step | Mirrors GHA job | Queue (machine) | When |
|---|---|---|---|
| `pre-commit` | `pre-commit` gate | `small_cpu_queue_premerge` (r6in.large) | every build |
| `plugin-contracts` | `e2e-test-plugin-contracts` | `medium_cpu_queue_premerge` (r6in.4xlarge) | every build |
| `agent-adapter` | `agent-adapter-test` | `small_cpu_queue_premerge` | PR / manual builds |
| `unit` | `e2e-test-unit` | `medium_cpu_queue_premerge` | PR / manual builds |

All non-gate steps `depend_on` the pre-commit gate, matching the GHA
`needs: pre-commit`. Suites run their test files sequentially inside one step
because these queues boot a fresh EC2 instance per job — a per-file matrix
would be mostly boot + pip-install time. The `unit` step pulls
`inferactinc/public:vime-latest` on every build (no local image cache on
ephemeral instances); if pull time becomes a problem, mirror the image to ECR
(the premerge queues already have read-only ECR access).

## Creating the pipeline (one-time, Buildkite UI)

Org `vllm`, cluster **CI** (the premerge queues live there).

1. New pipeline: name `vime-ci`, repository
   `https://github.com/vllm-project/vime.git`.
2. Steps — paste this instead of the default bare upload step (saves one
   bootstrap instance boot per build):

   ```yaml
   steps:
     - label: ":pipeline: generate"
       command: python3 .buildkite/generate_pipeline.py | buildkite-agent pipeline upload
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

## Event behaviour

Buildkite step conditionals can't see GitHub event details, so the generator
decides what to emit from `BUILDKITE_BRANCH` / `BUILDKITE_PULL_REQUEST`:

- PR build or manual branch build → all four steps.
- Push to `main` → `pre-commit` + `plugin-contracts` only (same reduced set
  as GHA, which keeps main-push builds cheap while catching PR-pair
  regressions).

Test the generator locally:

```bash
python3 .buildkite/generate_pipeline.py                        # manual-build view
BUILDKITE_BRANCH=main python3 .buildkite/generate_pipeline.py  # main-push view
```
