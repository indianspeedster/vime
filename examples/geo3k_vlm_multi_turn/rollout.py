from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from vime.rollout.vllm_rollout import (
    GenerateState,
    _build_inference_sampling_params,
    _inference_generate_tokens_and_logprobs,
    _mm_render_response_to_generate_body,
)
from vime.utils.http_utils import post
from vime.utils.processing_utils import build_processor_kwargs, encode_image_for_rollout_engine
from vime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"

# Dummy messages used for calculating trim length in chat template encoding.
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


def _content_to_render_format(content: list[dict]) -> list[dict]:
    """Convert env-style message content (image objects) to render-route format (image_url data URLs)."""
    out: list[dict] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "image" and part.get("image") is not None:
            data_url = encode_image_for_rollout_engine(part["image"])
            out.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            out.append(part)
    return out


def _build_initial_user_message(sample: Sample) -> dict:
    """Build the initial user message for the render route (same layout as vllm_rollout MM render)."""
    images = (sample.multimodal_inputs or {}).get("images") or []
    if not images:
        return {"role": "user", "content": sample.prompt}
    content: list[dict] = [{"type": "text", "text": sample.prompt}]
    for image in images:
        content.append({"type": "image", "image": image})
    return {"role": "user", "content": content}


def _messages_for_render(messages: list[dict]) -> list[dict]:
    """Normalize per-turn messages to the render-route shape (image -> image_url)."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            out.append({"role": msg["role"], "content": _content_to_render_format(content)})
        else:
            out.append(msg)
    return out


def _prepare_initial_inputs(sample: Sample, processor, tokenizer) -> tuple[list[int], dict | None]:
    """Initial train-side features from dataset-rendered ``sample.prompt`` (no re-template)."""
    multimodal_train_inputs = None
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal = any(value is not None for value in raw_multimodal_inputs.values())
    if processor and has_multimodal:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    return list(prompt_ids), multimodal_train_inputs


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
) -> tuple[list[int], dict | None, dict | None]:
    """Encode a fresh env observation message (may include images/videos)."""
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}
    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        if images or videos:
            multimodal_inputs = {"images": images, "videos": videos}
            processor_output = processor(text=formatted_prompt, **build_processor_kwargs(multimodal_inputs))
            prompt_ids = processor_output["input_ids"][0]
            multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
            } or None
        else:
            prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    return list(prompt_ids), multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """Concatenate per-turn processor outputs along dim 0 for torch tensor fields."""
    if not chunks:
        return None
    values_by_key: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)
    merged: dict = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
    return merged


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
    *,
    track_response: bool = False,
) -> None:
    sample.tokens.extend(tokens_to_add)
    if track_response:
        response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    if track_response:
        sample.response_length = len(response_tokens)


def _append_rendered_prefix(
    sample: Sample,
    response_tokens: list[int],
    rendered_ids: list[int],
) -> int:
    """Append render delta with ``loss_mask=0``; fail on prefix drift."""
    existing = list(sample.tokens)
    if len(rendered_ids) < len(existing):
        raise ValueError(
            f"render token_ids length ({len(rendered_ids)}) shorter than sample.tokens ({len(existing)})"
        )
    if rendered_ids[: len(existing)] != existing:
        raise ValueError("render token_ids prefix mismatch with sample.tokens (chat template drift)")
    delta = rendered_ids[len(existing) :]
    if delta:
        _append_to_sample(sample, response_tokens, delta, [0.0] * len(delta), loss_mask_val=0, track_response=False)
    return len(delta)


def _update_multimodal_state(
    sample: Sample,
    obs_multimodal_inputs: dict | None,
    obs_multimodal_train_inputs: dict | None,
    multimodal_train_inputs_buffer: list[dict | None],
) -> None:
    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
                else:
                    sample.multimodal_inputs[key] = val
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)


async def _render_messages(args, base_url: str, messages: list[dict]) -> tuple[dict, list[int]]:
    """Call ``/v1/chat/completions/render`` and return generate body + prompt token ids."""
    render_payload = {"model": args.hf_checkpoint, "messages": _messages_for_render(messages)}
    render_data = await post(f"{base_url}/v1/chat/completions/render", render_payload)
    body = _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)
    token_ids = body["token_ids"]
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return body, [int(x) for x in token_ids]


async def _generate_from_render_body(base_url: str, body: dict, sampling_params: dict) -> dict:
    generate_body = dict(body)
    generate_body["sampling_params"] = sampling_params
    return await post(f"{base_url}/inference/v1/generate", generate_body)


def _process_env_step(
    env: BaseInteractionEnv,
    response_text: str,
    tokenizer,
    processor,
    args: Any,
    sample_metadata: dict,
) -> tuple[dict | None, list[int] | None, dict | None, dict | None, bool]:
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    obs_prompt_ids, obs_multimodal_inputs, obs_multimodal_train_inputs = _encode_observation_for_generation(
        tokenizer,
        processor,
        next_user_message,
        sample_metadata,
        getattr(args, "apply_chat_template", False),
        getattr(args, "apply_chat_template_kwargs", None),
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return next_user_message, obs_prompt_ids, obs_multimodal_inputs, obs_multimodal_train_inputs, False


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment via the vLLM render route."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    if args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

    env_module = _load_env_module(args.rollout_interaction_env_path)
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)

    prompt_ids, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = []
    response_tokens: list[int] = []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    messages: list[dict] = [_build_initial_user_message(sample)]
    sampling_params = sampling_params.copy()
    inference_sampling_params = _build_inference_sampling_params(sampling_params)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)

    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        for turn_idx in range(args.max_turns):
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if budget is not None:
                inference_sampling_params["max_tokens"] = budget

            render_body, rendered_ids = await _render_messages(args, base_url, messages)
            prefix_added = _append_rendered_prefix(sample, response_tokens, rendered_ids)
            if budget is not None:
                budget -= prefix_added
                if budget <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break
                inference_sampling_params["max_tokens"] = budget

            output = await _generate_from_render_body(base_url, render_body, inference_sampling_params)
            choice = output["choices"][0]
            finish_reason = choice.get("finish_reason") or "stop"
            new_tokens, new_logprobs = _inference_generate_tokens_and_logprobs(choice)

            if not new_tokens:
                if finish_reason in ("abort", "cancelled"):
                    sample.status = Sample.Status.ABORTED
                    break

            response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
            _append_to_sample(
                sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1, track_response=True
            )
            if budget is not None:
                budget -= len(new_tokens)

            messages.append({"role": "assistant", "content": response_text})

            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_reason in ("abort", "cancelled"):
                sample.status = Sample.Status.ABORTED
                break

            next_user_message, obs_prompt_ids, obs_multimodal_inputs, obs_multimodal_train_inputs, done = (
                _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata)
            )
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            assert obs_prompt_ids is not None
            _append_to_sample(
                sample,
                response_tokens,
                obs_prompt_ids,
                [0.0] * len(obs_prompt_ids),
                loss_mask_val=0,
                track_response=False,
            )
            if budget is not None:
                budget -= len(obs_prompt_ids)

            _update_multimodal_state(
                sample, obs_multimodal_inputs, obs_multimodal_train_inputs, multimodal_train_inputs_buffer
            )
            messages.append(next_user_message)

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= args.max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
