# Tau-Bench Multi-Turn Tool Use

Multi-turn tool-use RL training in [tau-bench](https://github.com/JD-ETH/tau-bench) environments with vime vLLM rollout.

## Setup

```bash
pip install -e /path/to/vime --no-deps --no-build-isolation

git clone https://github.com/JD-ETH/tau-bench.git /path/to/tau-bench-src
cd /path/to/tau-bench-src && git checkout feature/litellm-retry
pip install -e . --no-deps && pip install litellm

cd /path/to/vime/examples/tau-bench
python tau1_mock.py --local_dir /path/to/datasets/tau-bench/
```

Model (Qwen3-4B-Instruct-2507):

```bash
export MODEL_ARGS_ROTARY_BASE=5000000
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /path/to/Qwen3-4B-Instruct-2507 \
  --save /path/to/Qwen3-4B-Instruct-2507_torch_dist
```

## Run

```bash
cd /path/to/vime
bash examples/tau-bench/run_qwen3_4B.sh
```

Key flags (set in `run_qwen3_4B.sh`):

- `--custom-generate-function-path generate_with_tau.generate`
- `--custom-rm-path generate_with_tau.batched_tau_bench_rm`
- Ray `PYTHONPATH` must include **`examples/tau-bench`** (before vime root)
- `--vllm-max-model-len 16384` and `--rollout-max-context-len 16384`
- `unset PYTORCH_CUDA_ALLOC_CONF` before colocate / non-colocate train

Default tau settings (formerly in yaml) are applied inside `generate_with_tau._ensure_tau_args`:

- `max_turns=10`, `tau_env=retail`, local vLLM user sim via `openai/local-qwen3-4b`

To use Gemini as user simulator, pass overrides on the train command line, e.g.:

```bash
--tau-user-model gemini-2.0-flash-lite --tau-user-model-provider gemini
```

(Requires `GEMINI_API_KEY` in the Ray runtime env.)
