"""Coding-agent sandbox helpers.

The provider-agnostic sandbox contract and E2B backend live in
``vime.agent.sandbox``. This module keeps the coding-agent/SWE-specific
bootstrap, Claude Code runner, diff capture, and fresh-sandbox evaluator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import lzma
import os
import shlex
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from vime.agent.sandbox import E2BSandbox, ModalSandbox, Sandbox


logger = logging.getLogger(__name__)

# Paths inside the sandbox (avoid clashes with image-shipped paths).
_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_SWEPRO_DIR = "/workspace/swepro_eval"

SWE_HOST_NODE_TARBALL = Path(
    os.environ.get(
        "SWE_HOST_NODE_TARBALL",
        "/path/to/node-v22.20.0-linux-x64.tar.xz",
    )
)
SWE_HOST_CC_TARBALL = Path(
    os.environ.get(
        "SWE_HOST_CC_TARBALL",
        "/path/to/anthropic-ai-claude-code.tgz",
    )
)
SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
CC_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)

_BOOT_SEM: asyncio.Semaphore | None = None
_ADAPTER_SEM: asyncio.Semaphore | None = None


def make_sandbox(image: str) -> Sandbox:
    """Construct the configured sandbox backend for ``image``.

    Backend is chosen by ``VIME_AGENT_SANDBOX_BACKEND`` (read per call so run.sh
    / tests can set it): ``e2b`` (default, unchanged behavior) or ``modal``.
    Both backends satisfy the same :class:`Sandbox` Protocol, so the work-sandbox
    and eval-sandbox call sites below stay backend-agnostic.
    """
    backend = os.environ.get("VIME_AGENT_SANDBOX_BACKEND", "e2b").strip().lower()
    if backend == "modal":
        return ModalSandbox(image)
    if backend in ("", "e2b"):
        return E2BSandbox(image)
    raise ValueError(f"unknown VIME_AGENT_SANDBOX_BACKEND={backend!r} (expected 'e2b' or 'modal')")


# ---------------------------------------------------------------------------
# Sandbox bootstrap (Node + Claude Code + agent user)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def boot_agent_sandbox(image: str) -> AsyncIterator[Sandbox]:
    """Boot a fresh sandbox (E2B or Modal) and install the Claude Code toolchain.

    This is the provisioning wrapper for the work sandbox: create the sandbox
    from the dataset image, install Node 22 + Claude Code CLI from host
    tarballs, retry transient boot/install failures, and close the sandbox when
    the caller leaves the context.
    """
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)

    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = make_sandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await install_node22(cand, SWE_HOST_NODE_TARBALL)
                    await install_claude_code(cand, SWE_HOST_CC_TARBALL)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[coding_agent_rl] provision attempt %d/%d failed: %s: %s",
                attempt + 1,
                SWE_BOOT_RETRIES,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def install_node22(sb: Sandbox, host_tarball: Path) -> None:
    """Node 22 over the base image (Debian 12 ships 16; cli.js needs >= 20).
    Decompresses .xz on the host (cached) so sandboxes without xz-utils can
    still run plain `tar xf`. npm prefix=/usr/local required for sweap-images."""
    host_tarball = Path(host_tarball)
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"coding_agent_rl.{host_tarball.stem}.tar"
        if not plain.exists():
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(host_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        host_tarball = plain
    await sb.write_file("/tmp/node22.tar", host_tarball)
    await sb.exec(
        "set -e && mkdir -p /opt/node22 && "
        "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
        "hash -r 2>/dev/null || true && node --version && npm --version",
        user="root",
        timeout=180,
        check=True,
    )


async def install_claude_code(sb: Sandbox, host_tarball: Path) -> None:
    await sb.write_file("/tmp/claude-code.tgz", host_tarball)
    await sb.exec(
        "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/claude-code.tgz "
        "&& ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        user="root",
        timeout=300,
        check=True,
    )


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff.
    Settings file pre-acks bypass-permissions so claude-code starts headless."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent && "
        f"mkdir -p /home/agent/.claude && "
        f'echo \'{{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}}\' '
        f"| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
        f"chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
        user="root",
        check=True,
        timeout=60,
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd") if swepro else None
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Uni-agent mode: str_replace_editor / execute_bash / submit loop
# ---------------------------------------------------------------------------

_TOOLS_DIR = Path(__file__).parent / "tools"

UNIAGENT_SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer to solve tasks."

_cf_id = os.environ.get("CF_ACCESS_CLIENT_ID")
_cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
_CF_ACCESS_HEADERS = {"CF-Access-Client-Id": _cf_id, "CF-Access-Client-Secret": _cf_secret} if _cf_id else None

_UNIAGENT_TOOL_SCHEMAS = [
    {
        "name": "str_replace_editor",
        "description": (
            "Custom editing tool for viewing, creating and editing files. "
            "* `view` shows a file (with line numbers) or directory listing. "
            "* `str_replace` replaces an exact substring in a file. "
            "* `create` writes a new file. "
            "* `insert` inserts text after a given line number. "
            "* `undo_edit` reverts the last edit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The sub-command to run.",
                },
                "path": {"type": "string", "description": "Absolute path to a file or directory."},
                "file_text": {"type": "string", "description": "Content for `create`."},
                "old_str": {"type": "string", "description": "Exact text to replace for `str_replace`."},
                "new_str": {"type": "string", "description": "Replacement text (or text to insert for `insert`)."},
                "insert_line": {"type": "integer", "description": "Line number after which to insert for `insert`."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[start_line, end_line] for `view` with a range.",
                },
            },
            "required": ["command", "path"],
        },
    },
    {
        "name": "execute_bash",
        "description": "Execute a bash command in the terminal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "submit",
        "description": "Signal that the task is complete and the fix is ready for evaluation.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

UNIAGENT_SWE_PROMPT = (
    "<uploaded_files>\n{workdir}\n</uploaded_files>\n"
    "I have uploaded a python code repository in the {workdir} directory. "
    "You can explore and modify files using the available tools. "
    "Consider the following issue description:\n\n"
    "<issue_description>\n{problem_statement}\n</issue_description>\n\n"
    "Can you help me implement the necessary changes to the repository to fix the <issue_description>?\n"
    "I have already taken care of all changes to any of the test files described in the "
    "<issue_description>. This means you DON'T have to modify the testing logic or any of the "
    "tests in any way!\n"
    "Also the development Python environment is already set up for you (i.e., all dependencies "
    "already installed), so you don't need to install other packages.\n"
    "Your task is to make the minimal changes to non-test files in the {workdir} directory to "
    "ensure the <issue_description> is satisfied.\n\n"
    "Follow these steps to resolve the issue:\n"
    "1. First, explore the codebase to locate and understand the code relevant to the "
    "<issue_description>.\n"
    "2. Assess whether you can reproduce the issue:\n"
    "   - Create a script at '{workdir}/reproduce_issue.py' that demonstrates the error.\n"
    "   - Execute this script to confirm the error behavior.\n"
    "3. Analyze the root cause and implement your fix (minimal changes to non-test files).\n"
    "4. Verify your solution by rerunning the reproduction script.\n"
    "5. Run the relevant unit tests to confirm your fix passes and has no regressions.\n"
    "6. When done, call the `submit` tool."
)


def _build_str_replace_cmd(tool_input: dict) -> str:
    command = tool_input.get("command", "view")
    path = tool_input.get("path", "")
    parts = ["str_replace_editor", command, "--path", path]
    for key, flag in [("old_str", "--old_str"), ("new_str", "--new_str"), ("file_text", "--file_text")]:
        val = tool_input.get(key)
        if val is not None:
            parts += [flag, val]
    if tool_input.get("insert_line") is not None:
        parts += ["--insert_line", str(tool_input["insert_line"])]
    if tool_input.get("view_range") is not None:
        vr = tool_input["view_range"]
        parts += ["--view_range", f"[{vr[0]}, {vr[1]}]"]
    return shlex.join(parts)


async def _install_uniagent_tools(sb: Sandbox) -> None:
    for name in ("str_replace_editor", "execute_bash", "submit"):
        host_path = _TOOLS_DIR / name
        await sb.write_file(f"/usr/local/bin/{name}", host_path, user="root")
        await sb.exec(f"chmod +x /usr/local/bin/{name}", user="root", timeout=10, check=True)
    await sb.exec(
        "pip install 'tree-sitter==0.21.3' tree-sitter-languages -q 2>/dev/null || true",
        user="root", timeout=120, check=False,
    )


@asynccontextmanager
async def boot_uniagent_sandbox(image: str) -> AsyncIterator[Sandbox]:
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)
    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = make_sandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning("[uniagent] boot attempt %d/%d failed: %s", attempt + 1, SWE_BOOT_RETRIES, e)
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def run_uniagent_loop(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
    problem_statement: str = "",
    swepro: dict | None = None,
    pre_commands: list[str] | str | None = None,
    max_turns: int = 300,
    action_timeout: int = 300,
) -> None:
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required for uniagent mode; pip install httpx")

    await ensure_agent_user(sb, workdir)
    if swepro:
        await apply_before_repo_set_cmd(sb, workdir, swepro)
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await _install_uniagent_tools(sb)

    prompt = UNIAGENT_SWE_PROMPT.replace("{workdir}", workdir).replace(
        "{problem_statement}", problem_statement or ""
    )
    messages: list[dict] = [{"role": "user", "content": prompt}]
    deadline = time.time() + time_budget_sec
    logger.info("[uniagent] %s: starting loop (max_turns=%d budget=%ds)", session_id, max_turns, time_budget_sec)

    extra_headers = {"Authorization": f"Bearer {session_id}"}
    if _CF_ACCESS_HEADERS:
        extra_headers.update(_CF_ACCESS_HEADERS)

    no_tool_streak = 0
    total_infer_sec = 0.0
    total_tool_sec = 0.0
    total_sem_wait_sec = 0.0
    global _ADAPTER_SEM
    if _ADAPTER_SEM is None:
        _ADAPTER_SEM = asyncio.Semaphore(int(os.environ.get("SWE_ADAPTER_CONCURRENCY", "80")))
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=60.0, read=600.0, write=60.0, pool=120.0)) as client:
        for turn in range(max_turns):
            if time.time() > deadline:
                logger.info("[uniagent] %s: time budget hit at turn %d", session_id, turn)
                break

            data = None
            t_sem = time.time()
            for attempt in range(12):
                try:
                    async with _ADAPTER_SEM:
                        t_sem_acquired = time.time()
                        if attempt == 0:
                            total_sem_wait_sec += t_sem_acquired - t_sem
                        t_infer = time.time()
                        resp = await client.post(
                            f"{adapter_url}/v1/messages",
                            json={
                                "model": "vime-actor",
                                "max_tokens": 16384,
                                "system": UNIAGENT_SYSTEM_PROMPT,
                                "messages": messages,
                                "tools": _UNIAGENT_TOOL_SCHEMAS,
                            },
                            headers=extra_headers,
                        )
                        total_infer_sec += time.time() - t_infer
                    if resp.status_code in (429, 500, 502, 503, 504):
                        backoff = min(2 ** attempt, 30) + (attempt * 0.3)
                        logger.warning("[uniagent] %s: turn %d attempt %d got %d; backoff %.1fs",
                                       session_id, turn, attempt, resp.status_code, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    backoff = min(2 ** attempt, 30)
                    logger.warning("[uniagent] %s: turn %d attempt %d err %s; backoff %ss",
                                   session_id, turn, attempt, str(e)[:120], backoff)
                    await asyncio.sleep(backoff)
            if data is None:
                logger.warning("[uniagent] %s: turn %d exhausted retries, ending", session_id, turn)
                break

            content = data.get("content", [])
            stop_reason = data.get("stop_reason", "")
            messages.append({"role": "assistant", "content": content})

            tool_uses = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
            if turn == 0:
                types = [c.get("type") for c in content if isinstance(c, dict)]
                logger.info("[uniagent] %s: turn0 stop=%s block_types=%s n_tooluse=%d",
                            session_id, stop_reason, types, len(tool_uses))
            if not tool_uses:
                no_tool_streak += 1
                if no_tool_streak >= 5:
                    logger.info("[uniagent] %s: %d consecutive no-tool turns, ending", session_id, no_tool_streak)
                    break
                messages.append({
                    "role": "user",
                    "content": "Please proceed by calling one of the available tools "
                               "(str_replace_editor / execute_bash / submit). Do not reply with prose only.",
                })
                continue
            no_tool_streak = 0

            if any(tc.get("name") == "submit" for tc in tool_uses):
                logger.info("[uniagent] %s: submit at turn %d", session_id, turn)
                break

            t_tools = time.time()
            tool_results = []
            for tc in tool_uses:
                tc_id = tc.get("id", f"tool_{turn}_{len(tool_results)}")
                tc_name = tc.get("name", "")
                tc_input = tc.get("input", {})
                if tc_name == "execute_bash":
                    cmd = tc_input.get("command", "echo '(empty)'")
                    rc, out, err = await sb.exec(cmd, user="root", timeout=action_timeout, check=False)
                    output = ((out or "") + (err or ""))[:10000]
                elif tc_name == "str_replace_editor":
                    cmd = _build_str_replace_cmd(tc_input)
                    rc, out, err = await sb.exec(cmd, user="root", timeout=30, check=False)
                    output = ((out or "") + (err or ""))[:16000]
                else:
                    output = f"Unknown tool: {tc_name}"
                tool_results.append({"type": "tool_result", "tool_use_id": tc_id, "content": output})
            total_tool_sec += time.time() - t_tools

            messages.append({"role": "user", "content": tool_results})

    n_turns = len([m for m in messages if m["role"] == "assistant"])
    elapsed_total = time.time() - (deadline - time_budget_sec)
    logger.info(
        "[uniagent] %s: loop finished turns=%d elapsed=%.0fs infer=%.1fs(%.0f%%) tool=%.1fs(%.0f%%) sem_wait=%.1fs(%.0f%%) avg_infer=%.1fs avg_tool=%.1fs",
        session_id, n_turns, elapsed_total,
        total_infer_sec, 100 * total_infer_sec / max(elapsed_total, 1),
        total_tool_sec, 100 * total_tool_sec / max(elapsed_total, 1),
        total_sem_wait_sec, 100 * total_sem_wait_sec / max(elapsed_total, 1),
        total_infer_sec / max(n_turns, 1),
        total_tool_sec / max(n_turns, 1),
    )


# ---------------------------------------------------------------------------
# Agent run (workspace prep + claude-code spawn + done-marker poll)
# ---------------------------------------------------------------------------
async def run_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
    problem_statement: str = "",
    swepro: dict | None = None,
    pre_commands: list[str] | str | None = None,
    prompt: str | None = None,
) -> int:
    """Prepare the SWE workspace, write PROBLEM_STATEMENT.md, then run CC."""
    await ensure_agent_user(sb, workdir)
    if swepro:
        await apply_before_repo_set_cmd(sb, workdir, swepro)
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        problem_statement or "",
        user="agent",
    )
    return await _spawn_claude_code(
        sb,
        workdir=workdir,
        session_id=session_id,
        adapter_url=adapter_url,
        prompt=prompt or CC_PROMPT,
        time_budget_sec=time_budget_sec,
    )


async def _spawn_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    prompt: str,
    time_budget_sec: int,
) -> int:
    """Spawn claude-code detached + poll a done-marker file.

    E2B's gateway resets HTTP/2 around 6.5 min, so we can't keep a long-lived
    foreground exec. The launcher writes the exit code into a marker file
    and we poll it every 5s via short RPCs (which also keeps the sandbox
    alive against idle GC)."""
    done = f"{workdir}/.cagent_done"
    launcher = f"{workdir}/.cagent_run.sh"
    traj = f"{workdir}/claude_code_trajectory.jsonl"

    launcher_body = (
        "#!/bin/bash\n"
        f"cd {workdir}\n"
        "export HOME=/home/agent\n"
        f"/usr/local/bin/claude -p {json.dumps(prompt)} "
        f"--permission-mode bypassPermissions "
        f"--output-format stream-json --include-partial-messages "
        f"--include-hook-events --verbose "
        f"{os.environ.get('SWE_CLAUDE_EXTRA_ARGS', '').strip()} "
        f"2>&1 | tee {shlex.quote(traj)}\n"
        f"echo $? > {done}\n"
    )
    await sb.write_file(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {launcher}", user="agent", timeout=30)

    env = {
        "ANTHROPIC_BASE_URL": adapter_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": "vime-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    env_keys = ",".join(env.keys())
    await sb.exec(
        f"runuser -u agent --whitelist-environment={env_keys}"
        f" -- bash -c 'setsid {launcher} < /dev/null > /dev/null 2>&1 &'",
        user="root",
        env=env,
        timeout=30,
        check=True,
    )

    deadline = time.time() + time_budget_sec
    exit_code = -2  # convention: -2 = budget exceeded
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(
            f"test -f {done} && cat {done}",
            user="agent",
            timeout=15,
            check=False,
        )
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    return exit_code


async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        f"git diff -- . ':(exclude)PROBLEM_STATEMENT.md' "
        f"':(exclude)claude_code_trajectory.jsonl' "
        f"':(exclude).cagent_done' ':(exclude).cagent_run.sh'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


# ---------------------------------------------------------------------------
# Eval (fresh sandbox, apply diff, run dataset tests)
# ---------------------------------------------------------------------------
async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swepro: dict | None = None,
    eval_cmd: str | None = None,
    swebench_metadata: dict | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool, bool]:
    """Returns (reward, solved, applied_cleanly).

    No-test-cheating guarantee: the eval sandbox is built from the same image
    but starts CLEAN, so only the model-produced diff affects reward."""
    if not (swepro or eval_cmd or swebench_metadata):
        logger.warning("[e2b.evaluate] no swepro/eval_cmd/swebench_metadata; reward=0")
        return 0.0, False, True

    async with make_sandbox(image) as ev:
        await ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            logger.warning("[evaluate] %s: diff apply FAILED (%d chars)", swebench_metadata.get("instance_id", "?") if swebench_metadata else "?", len(diff_text or ""))
            return 0.0, False, False
        logger.info("[evaluate] %s: diff applied OK (%d chars)", swebench_metadata.get("instance_id", "?") if swebench_metadata else "?", len(diff_text or ""))

        if swepro:
            r, s = await _run_swepro(ev, workdir, swepro, timeout_sec)
            return r, s, True
        if swebench_metadata:
            r, s = await _run_swebench_eval(ev, workdir, swebench_metadata, timeout_sec)
            return r, s, True
        r, s = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        return r, s, True


async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            text = Path(host_p).read_text()
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", text, user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


async def apply_pre_commands(ev: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called by generate.py to keep the work sandbox baseline
    # aligned with eval (sweb-style pre_commands typically `git checkout
    # <base_sha> -f`, so skipping in work sandbox makes the model's diff
    # context mismatch the eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await ev.write_file(_PRE, "set -e\n" + body, user="agent")
    await ev.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user="agent")
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, _, _ = await ev.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh "
        f"{json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user="agent",
        check=False,
        timeout=120,
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return (1.0 if solved else 0.0), solved


async def _run_swebench_eval(ev: Sandbox, workdir: str, metadata: dict, timeout: int) -> tuple[float, bool]:
    """SWE-bench harness grading — inline, matching the 0611 eval that scored 71.6%."""
    import json as _json
    import re as _re
    from swebench.harness.constants import (
        END_TEST_OUTPUT, FAIL_ONLY_REPOS, MAP_REPO_VERSION_TO_SPECS,
        START_TEST_OUTPUT, EvalType, ResolvedStatus,
    )
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.python import get_test_directives
    from swebench.harness.utils import get_modified_files

    repo = metadata["repo"]
    version = metadata["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    test_patch = metadata["test_patch"]
    base_commit = metadata.get("base_commit", "")
    env_name = "testbed"
    repo_dir = workdir

    test_cmd = specs["test_cmd"]
    directives = get_test_directives(metadata)
    test_command = " ".join([test_cmd, *directives])

    _HEREDOC = "EOF_114329324912"
    apply_test_patch = f"git apply -v - <<'{_HEREDOC}'\n{test_patch}\n{_HEREDOC}"

    test_patch_files = get_modified_files(test_patch)
    if base_commit and test_patch_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_patch_files)}"
    else:
        reset_tests_command = "echo 'skip reset_tests (no base_commit or test_patch_files)'"

    eval_cmds = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_dir}",
    ]
    if "eval_commands" in specs:
        eval_cmds += specs["eval_commands"]
    eval_cmds += [
        f"git config --global --add safe.directory {repo_dir}",
        f"cd {repo_dir}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_cmds.append(specs["install"])
    eval_cmds += [
        reset_tests_command,
        apply_test_patch,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    eval_script = "#!/bin/bash\nset -uxo pipefail\n" + "\n".join(eval_cmds) + "\n"

    await ev.write_file("/tmp/_swebench_eval.sh", eval_script, user="root")
    _, stdout, _ = await ev.exec("bash /tmp/_swebench_eval.sh 2>&1", user="root", check=False, timeout=timeout)
    output = _re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout or "")

    # Log eval output size and key markers for debugging
    logger.info("[swebench_eval] %s: eval output %d chars, has_start=%s, has_end=%s, reset_cmd=%s",
                metadata["instance_id"], len(output),
                START_TEST_OUTPUT in output, END_TEST_OUTPUT in output,
                reset_tests_command[:60])

    solved = False
    if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
        test_content = output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        status_map = MAP_REPO_TO_PARSER[repo](test_content, None)
        eval_ref = {
            "instance_id": metadata["instance_id"],
            "FAIL_TO_PASS": _json.loads(metadata["FAIL_TO_PASS"]) if isinstance(metadata["FAIL_TO_PASS"], str) else metadata["FAIL_TO_PASS"],
            "PASS_TO_PASS": _json.loads(metadata["PASS_TO_PASS"]) if isinstance(metadata["PASS_TO_PASS"], str) else metadata["PASS_TO_PASS"],
        }
        f2p = eval_ref["FAIL_TO_PASS"]
        p2p = eval_ref["PASS_TO_PASS"]
        eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        resolved = get_resolution_status(report)
        solved = resolved == ResolvedStatus.FULL.value
        # Log FAIL_TO_PASS detail for failed instances
        f2p_report = report.get("FAIL_TO_PASS", {})
        f2p_success = len(f2p_report.get("success", []))
        f2p_failure = len(f2p_report.get("failure", []))
        p2p_report = report.get("PASS_TO_PASS", {})
        p2p_success = len(p2p_report.get("success", []))
        p2p_failure = len(p2p_report.get("failure", []))
        logger.info("[swebench_eval] %s: resolved=%s F2P=%d/%d P2P=%d/%d report=%s",
                    metadata["instance_id"], solved,
                    f2p_success, f2p_success + f2p_failure,
                    p2p_success, p2p_success + p2p_failure,
                    _json.dumps(report, default=str)[:300])
    else:
        logger.warning("[swebench_eval] %s: test output markers not found, output_tail=%s",
                       metadata["instance_id"], output[-200:] if output else "EMPTY")

    return (1.0 if solved else 0.0), solved


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> tuple[float, bool]:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0
