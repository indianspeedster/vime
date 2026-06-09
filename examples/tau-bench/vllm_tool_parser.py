from typing import Any

from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


class _StubTokenizer:
    """Hermes2ProToolParser requires a tokenizer at init but does not use it in extract_tool_calls."""

    pass


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25"):
    """
    This function mimics the function call parser API from
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L952
    But running locally, using vLLM's Hermes2ProToolParser (handles the same
    <tool_call>...</tool_call> format used by Qwen2.5).
    """
    tool_parser = Hermes2ProToolParser(_StubTokenizer())
    result = tool_parser.extract_tool_calls(response, None)

    normal_text = result.content or ""
    calls = []
    if result.tool_calls:
        for tc in result.tool_calls:
            calls.append({"name": tc.function.name, "parameters": tc.function.arguments})

    return {
        "normal_text": normal_text,
        "calls": calls,
    }
