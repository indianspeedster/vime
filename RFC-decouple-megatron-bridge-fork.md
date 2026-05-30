# RFC: Decouple slime/vime from the `fzyzcjy/Megatron-Bridge` fork — migrate to upstream `NVIDIA-NeMo/Megatron-Bridge`

| | |
|---|---|
| **Status** | Draft / for discussion |
| **Author** | (you) |
| **Scope** | `slime_plugins/megatron_bridge/`, Dockerfile, `build_conda.sh` |
| **Affects** | the optional GLM-4.6V (`Glm4vMoeForConditionalGeneration`) bridge; the megatron-bridge dependency pin |
| **Date** | 2026-05-30 |

---

## 0. TL;DR

slime/vime is the only RL framework in the reference set (verl, SkyRL, OpenRLHF, NeMo-RL, ROLL, AReaL, prime-rl) that is **bound to a private fork of Megatron-Bridge** (`fzyzcjy/Megatron-Bridge@dev_rl`). The binding comes down to **a single import** in `slime_plugins/megatron_bridge/glm4v_moe.py`:

```python
from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider   # fork-only symbol
class Glm4vMoeVLModelProvider(Qwen3MoEModelProvider): ...
```

`Qwen3MoEModelProvider` is a **pure dataclass** (no methods, ~20 Qwen3‑MoE config defaults on top of the public `GPTModelProvider`) that exists **only** in the fork. Upstream and the `radixark` fork express the same config through `Qwen3MoEBridge.provider_bridge()` instead, and ship no such dataclass — so the import `ImportError`s on any non-fork build, which is the root cause of the `slime_plugins/megatron_bridge/__init__.py` import guard.

This RFC proposes to **remove the single fork dependency** (subclass the public `GPTModelProvider` and inline the defaults) and **standardize on upstream `NVIDIA-NeMo/Megatron-Bridge`**, matching verl/SkyRL. A source-level diff shows upstream's conversion core is **newer and more correct** than the currently-shipped `radixark` fork (notably for MoE / expert-parallel / FP8 conversion), so this is a net upgrade, not a regression.

---

## 1. Background

### 1.1 What Megatron-Bridge does for slime

`megatron.bridge` (Megatron-Bridge) converts weights and configs between HuggingFace and Megatron-Core ("mcore") formats. slime uses it on the public surface in several places:

- `slime/backends/megatron_utils/model.py`, `model_provider.py`, `checkpoint.py`, `update_weight/hf_weight_iterator_bridge.py` — all via `from megatron.bridge import AutoBridge` + `AutoBridge.from_hf_pretrained(...)`.
- `slime_plugins/megatron_bridge/glm4v_moe.py` — registers a **custom** GLM‑4.6V VL bridge through the public `MegatronModelBridge.register_bridge` / `MegatronMappingRegistry` / `AutoMapping` API.
- `slime/utils/megatron_bridge_utils.py` — a few small, defensive monkey-patches (`patch_auto_bridge_hf_config`, `patch_megatron_model`).

All of the above are **public** Megatron-Bridge API. The single exception is `glm4v_moe.py`'s use of `Qwen3MoEModelProvider`.

### 1.2 The three Megatron-Bridge variants in play

| variant | repo / ref | HEAD | last commit | role |
|---|---|---|---|---|
| **upstream (official)** | `NVIDIA-NeMo/Megatron-Bridge` | `118ff1a` | 2026-05-29 | the public package; what verl/SkyRL track |
| **radixark** | `radixark/Megatron-Bridge@bridge` | `6fde1c8` | 2026-05-26 | installed by slime/vime **Dockerfile** (`--no-deps`) |
| **fzyzcjy** | `fzyzcjy/Megatron-Bridge@dev_rl` | `35b4ebf` | 2026-01-20 | installed by slime **`build_conda.sh:51`** (conda path) |

**slime is internally inconsistent:** the conda path installs `fzyzcjy@dev_rl` (which *has* `Qwen3MoEModelProvider`), but the docker path installs `radixark@bridge` (which *does not*). On the r3 image (docker path), `glm4v_moe.py` therefore fails to import without the guard. `megatron-core` in the r3 image is `0.16.0rc0` (Megatron-LM `1dcf0da`).

---

## 2. Problem statement

1. **Fork lock-in.** slime/vime cannot run on the public `megatron-bridge` package because `glm4v_moe.py` imports a fork-only symbol. Every other RL framework runs on upstream.
2. **Inconsistency.** conda → `fzyzcjy@dev_rl`, docker → `radixark@bridge`. The two forks are different lineages with different feature sets; only one has the symbol `glm4v_moe` needs.
3. **Staleness risk.** `fzyzcjy@dev_rl` is a ~4-month-old upstream snapshot (Jan 2026). `radixark@bridge`'s conversion core is also an older upstream variant (see §4.3). Both miss upstream's newer models and bug fixes.
4. **A guard, not a fix.** The current `try/except ImportError` guard in `slime_plugins/megatron_bridge/__init__.py` only prevents the optional GLM‑4.6V bridge from taking down the whole bridge path; it does not let GLM‑4.6V actually run on a non-fork build.

---

## 3. Why other frameworks don't need a fork

Grepping `qwen_provider` / `Qwen3MoEModelProvider` across the reference set: **only slime (and vime, which vendored slime's plugin) matches.** verl / SkyRL / ROLL / OpenRLHF / AReaL / prime-rl / NeMo-RL are all empty. Three integration philosophies:

- **verl** — hand-written in-tree converters (`verl/models/mcore/{registry,weight_converter,config_converter}.py`, `scripts/converter_hf_to_mcore.py`). Uses `megatron.bridge` only for peripheral utilities (LoRA hooks, checkpoint, `AutoBridge`). Never inherits an internal provider dataclass → no fork.
- **SkyRL** — pins **upstream** `NVIDIA-NeMo/Megatron-Bridge@8382dc3` in `pyproject.toml`, uses the high-level `AutoBridge.from_hf_pretrained(...).to_megatron_provider()` + `CanonicalLoRA`. No internal-class subclassing → no fork.
- **slime / vime** — subclasses an internal provider **dataclass** (`Glm4vMoeVLModelProvider(Qwen3MoEModelProvider)`) to reuse Qwen3‑MoE config infra for GLM‑4.6V. That dataclass lives only in `fzyzcjy@dev_rl` → fork lock-in.

The conclusion: **this is not a Megatron capability gap; it is an extension-style choice.** slime picked "inherit an upstream-internal class," and only one fork exposes it.

---

## 4. Detailed analysis (source-verified)

All three forks were cloned and read at the exact commits above; the r3 image's `Megatron-LM 1dcf0da` (megatron-core `0.16.0rc0`) was also cloned and read.

### 4.1 The coupling is exactly one symbol

In `glm4v_moe.py`, every import other than `Qwen3MoEModelProvider` exists in upstream HEAD (verified): `MegatronModelBridge`, `MegatronMappingRegistry`, `AutoMapping`/`GatedMLPMapping`/`QKVMapping`/`ReplicatedMapping`, `hook_hf_module_setattr_for_tp_grad_sync`, `GPTModelProvider`, and `AutoBridge`. The other slime call-site that looked fork-specific, `from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import ...`, also exists in **all three** variants. So `Qwen3MoEModelProvider` is the sole blocker.

### 4.2 What `Qwen3MoEModelProvider` is, and how upstream expresses the same thing

- **fork (`fzyzcjy`)** `models/qwen/qwen_provider.py:363`:
  ```python
  @dataclass
  class Qwen3MoEModelProvider(GPTModelProvider):   # no methods; ~20 field defaults
      normalization = "RMSNorm"; qk_layernorm = True; add_qkv_bias = False
      gated_linear_unit = True; moe_grouped_gemm = True
      moe_router_load_balancing_type = "aux_loss"; moe_token_dispatcher_type = "alltoall"
      moe_permute_fusion = True; ...
  ```
- **upstream & radixark** ship **no** `qwen/qwen_provider.py`. They build the identical config in `models/qwen/qwen3_moe_bridge.py::Qwen3MoEBridge.provider_bridge()`:
  ```python
  def provider_bridge(self, hf_pretrained):
      provider = super().provider_bridge(hf_pretrained)   # generic GPTModelProvider
      provider.normalization = "RMSNorm"; provider.qk_layernorm = True
      provider.add_qkv_bias = False; provider.gated_linear_unit = True
      provider.moe_grouped_gemm = True; provider.moe_router_load_balancing_type = "aux_loss"
      provider.moe_token_dispatcher_type = "alltoall"; provider.moe_permute_fusion = True
      return provider
  ```
  The values are **line-for-line identical** to the fork's dataclass defaults.

**Important correction to an earlier internal claim:** upstream is *not* allergic to provider dataclasses — upstream HEAD ships ~34 of them (`gemma_provider`, `llama_nemotron_provider`, `glm5_provider`, `qwen3_vl_provider`, …). The gap is narrow and specific: **the Qwen *text* family (`qwen/`) has no provider dataclass upstream** — it is bridge-method-only. `fzyzcjy` added one; it never went upstream. That single fork-only dataclass is what `glm4v_moe` subclasses.

### 4.3 radixark vs upstream conversion core — upstream is newer/more correct

The Dockerfile's `radixark@bridge` differs from upstream mostly because **radixark's conversion core is an older upstream variant** (plus small radixark-specific deltas), not because it carries patches slime needs. Concretely, in `models/conversion/`:

- **`param_mapping.py`** — same class/function set, except **upstream adds** `QKVGMapping` + `merge_qkvg_weights`/`split_qkvg_weights` (gated-QKV) that radixark lacks. Upstream is ahead.
- **`model_bridge.py`** — the MoE expert-index renaming helper differs, and **upstream's is strictly safer**:
  - upstream `_update_grouped_expert_number`: anchored regex `\.{type}(\d+)(?=$|\.)`, handles grouped-experts **and** `local_experts.N` **and** dual-pool (vision) MoE, and **explicitly excludes quantizer buffers** (`.weight_quantizer._amax`) from expert renumbering.
  - radixark `_update_expert_number`: naive `param_name.split(".weight")[-1]` and a blanket `if ".weight" in param_name` that **would also match `.weight_quantizer`** — i.e. radixark has a latent FP8/quant-MoE renumbering bug that upstream already fixed.
  - → migrating here is an **upgrade** for MoE / EP / FP8 conversion.
- **`auto_bridge.py`** — same signatures; upstream adds `trust_remote_code` handling + more supported models. radixark's one genuine unique delta: `HFWeightTuple` carries a 3rd field `megatron_param_name` (a quantization hook). slime references this only in a **TODO** (`hf_weight_iterator_bridge.py:53`), so it is not yet consumed.

Net: the "≈375 changed lines" between radixark and upstream are dominated by **upstream being ahead**, not by radixark patches that vime depends on.

---

## 5. Proposal

**Standardize on upstream `NVIDIA-NeMo/Megatron-Bridge` (pin a commit, like SkyRL), and remove the single fork dependency in `glm4v_moe.py`.**

### 5.1 Code change (the only required one)

Rewrite `Glm4vMoeVLModelProvider` to subclass the **public** `GPTModelProvider` and carry the Qwen3‑MoE defaults itself:

```python
from megatron.bridge.models.gpt_provider import GPTModelProvider   # public, present in all variants

@dataclass
class Glm4vMoeVLModelProvider(GPTModelProvider):
    # —— Qwen3-MoE arch defaults (formerly inherited from the fork's Qwen3MoEModelProvider) ——
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    moe_grouped_gemm: bool = True
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    # —— GLM-4.6V-specific fields (unchanged) ——
    image_token_id: int = 151363
    mrope_section: list[int] = field(default_factory=lambda: [8, 12, 12])
    # ... (rest unchanged; provide() unchanged)
```

Because `Glm4vMoeBridge.provider_bridge()` already sets the per-model fields explicitly from the HF config, and the base contributed only field defaults (no methods), this is mechanical. After it, `glm4v_moe.py` depends only on public symbols → imports cleanly on upstream / radixark / fzyzcjy, and the `__init__.py` import guard can be removed.

(Equivalent alternative: keep subclassing `GPTModelProvider` and set those defaults inside `provider_bridge()`, exactly as upstream's `Qwen3MoEBridge` does.)

### 5.2 Dependency change

- **Dockerfile**: replace `radixark/Megatron-Bridge@bridge` with a pinned upstream commit `NVIDIA-NeMo/Megatron-Bridge@<sha>` (install `--no-deps` as today so it uses the image's `megatron-core`).
- **`build_conda.sh`**: replace `fzyzcjy/Megatron-Bridge@dev_rl` with the **same** pinned upstream commit. This also removes the conda/docker inconsistency.

---

## 6. Risks & mitigations

| # | risk | severity | mitigation |
|---|---|---|---|
| 1 | GLM-4.6V import still fails on upstream until 5.1 lands | low (isolated, optional path) | do §5.1; keep guard until then |
| 2 | `megatron-core` version coupling — upstream HEAD may expect a core newer than `0.16.0rc0` | low–med | pin upstream to a commit whose core matches; build-time import check; radixark (3 days older) already works on 0.16 |
| 3 | quant path wants `HFWeightTuple.megatron_param_name` (radixark-only 3rd field) | low (only a TODO today) | contribute the field upstream, or carry a 1-line local patch when the quant TODO is implemented |
| 4 | checkpoint-format continuity — upstream's stricter expert-numbering could map a specific arch differently | med (verify, don't assume) | run one HF→mcore→HF conversion smoke test per MoE model vime trains; compare param-name mapping |
| 5 | slime monkey-patches (`patch_*`) target changed internals | low | they are `hasattr`-guarded and operate on public surfaces; smoke-test |

Note risk 4 is the only one needing real testing; everything else is a check or a known-small delta. Risk direction on the conversion core is **favorable** (upstream is the more-correct version, §4.3).

---

## 7. Alternatives considered

1. **Keep the `try/except` guard forever (status quo).** GLM‑4.6V remains unusable on non-fork builds; the inconsistency and staleness persist. Rejected as a non-fix.
2. **Install `fzyzcjy@dev_rl` in docker too** (make both paths use the fork). Removes the inconsistency but doubles down on a 4-month-stale private fork and keeps lock-in. Rejected.
3. **Upstream the `Qwen3MoEModelProvider` dataclass to `NVIDIA-NeMo`.** Clean long-term, but slower (depends on maintainers) and unnecessary — §5.1 removes the need entirely by using the public `GPTModelProvider`. Could still be done as a courtesy PR.
4. **Adopt verl's approach (own in-tree converters).** Largest change; discards Megatron-Bridge's value. Rejected.

---

## 8. Migration plan

1. **(code)** Apply §5.1 to `slime_plugins/megatron_bridge/glm4v_moe.py`; remove the `__init__.py` import guard.
2. **(verify, no fork)** In the r3 image, install **upstream** Megatron-Bridge `--no-deps`; run an import self-check of `slime_plugins.megatron_bridge` (should succeed without `Qwen3MoEModelProvider`).
3. **(verify, conversion)** Run one HF→mcore→HF smoke test for: (a) a Qwen3‑MoE model, (b) GLM‑4.x‑MoE, confirming identical param-name mappings vs the current radixark-based build (risk 4).
4. **(deps)** Pin Dockerfile + `build_conda.sh` to the same upstream commit; build-time import check (risk 2).
5. **(optional)** Open a courtesy PR to `NVIDIA-NeMo/Megatron-Bridge` adding a Qwen3‑MoE text provider dataclass, and/or the `HFWeightTuple.megatron_param_name` field, so future slime code needn't carry any local delta.

---

## 9. Open questions

- Which upstream commit to pin? (Prefer one whose bundled `megatron-core` matches the r3 image's `0.16.0rc0`, or bump Megatron-LM in lockstep.)
- Does any *other* vendored vime/slime model plugin (besides `glm4v_moe`) inherit a fork-only provider dataclass? (Audit `slime_plugins/` before pinning.)
- Is the quant path (the `megatron_param_name` TODO) on the near-term roadmap? If yes, plan the upstream contribution in step 5 accordingly.

---

## Appendix: evidence index

- Coupling: `slime_plugins/megatron_bridge/glm4v_moe.py:24` (import), `:478` (`class Glm4vMoeVLModelProvider(Qwen3MoEModelProvider)`).
- Fork-only symbol: `fzyzcjy/Megatron-Bridge@dev_rl:src/megatron/bridge/models/qwen/qwen_provider.py:363`. Absent in upstream `118ff1a` and radixark `6fde1c8`.
- Upstream equivalent: `…/models/qwen/qwen3_moe_bridge.py::Qwen3MoEBridge.provider_bridge()`.
- Public base: `megatron.bridge.models.gpt_provider.GPTModelProvider` (present in all three).
- Conversion-core diff: `models/conversion/{param_mapping,model_bridge,auto_bridge}.py` (radixark vs upstream).
- Install sites: `docker/Dockerfile` (radixark@bridge, `--no-deps`), `build_conda.sh:51` (fzyzcjy@dev_rl).
- Peer frameworks: `SkyRL/pyproject.toml` (NVIDIA-NeMo@8382dc3); verl `models/mcore/*` in-tree converters.
- Local clones used: `reference/Megatron-Bridge.{upstream,radixark,fzyzcjy}`, `reference/Megatron-LM-r3`.
