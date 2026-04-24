"""
Claude Code Proxy → opencode.ai
Convert Anthropic /v1/messages ↔ OpenAI chat/completions
"""

import json
import sys
import uuid
import time
import logging
import os
import sqlite3
import threading
import collections
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box

from config import API_KEY, PROXY, MODELS, ROUTES, get_model_config, HOST, PORT, WEB_PORT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# SQLite setup
_db_path = os.path.join(LOG_DIR, "requests.db")
_conn = sqlite3.connect(_db_path, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_db_lock = threading.Lock()
_conn.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        model TEXT NOT NULL,
        original_model TEXT,
        duration_ms INTEGER,
        tokens_input INTEGER,
        tokens_output INTEGER,
        tokens_cache INTEGER,
        success INTEGER,
        error TEXT
    )
""")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")

# Log buffer for Rich Live display
_log_lines = collections.deque(maxlen=200)
_LOG_VISIBLE = 35
_log_scroll = 0


def _log(msg: str):
    global _log_scroll
    # Skip internal API and uvicorn startup logs
    if "/api/" in msg or "Uvicorn running" in msg:
        return
    ts = time.strftime("%H:%M:%S")
    _log_lines.append(f"[{ts}] {msg}")
    # Auto-scroll to bottom on new log
    _log_scroll = max(0, len(_log_lines) - _LOG_VISIBLE)


def _save_request(req_id: str, model: str, original_model: str, duration_ms: int,
                  tokens_input: int, tokens_output: int, tokens_cache: int, success: bool = True, error: str = None):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        _conn.execute("""
            INSERT OR REPLACE INTO requests (id, timestamp, model, original_model, duration_ms,
                tokens_input, tokens_output, tokens_cache, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (req_id, timestamp, model, original_model, duration_ms,
              tokens_input, tokens_output, tokens_cache, 1 if success else 0, error))
        _conn.commit()


class _RichLogHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        # Skip internal API and uvicorn startup logs
        if "/api/" in msg or "Uvicorn running" in msg:
            return
        level = record.levelname
        ts = time.strftime("%H:%M:%S")
        _log_lines.append(f"[{ts}] [{level}] {msg}")


# Token usage tracking (in-memory, lost on restart)
_token_usage = {model: {"input": 0, "output": 0, "cache": 0} for model in MODELS}
_token_lock = threading.Lock()

# Shared HTTP client (reused across requests)
_transport = httpx.AsyncHTTPTransport(proxy=PROXY) if PROXY else None
_client = httpx.AsyncClient(transport=_transport, timeout=300)


@asynccontextmanager
async def lifespan(app):
    yield
    await _client.aclose()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _route_for(model_name: str) -> dict:
    name = model_name.lower()
    for r in ROUTES.values():
        if any(m in name for m in r["match"]):
            return r
    return ROUTES["sonnet"]


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for i in content:
            if isinstance(i, str):
                parts.append(i)
            elif isinstance(i, dict):
                if i.get("type") == "text":
                    parts.append(i.get("text", ""))
                elif i.get("type") == "thinking":
                    parts.append(i.get("thinking", ""))
                elif i.get("type") == "image":
                    parts.append(f"[image:{i.get('source', {}).get('type', 'unknown')}]")
                else:
                    parts.append(i.get("text", str(i)))
        return "\n".join(parts)
    return str(content) if content else ""


def anthropic_to_openai(body: dict, model: str) -> dict:
    thinking = isinstance(body.get("thinking"), dict) and body["thinking"].get("type") in ("enabled", "adaptive")

    messages = []

    # System prompt
    if system_text := _extract_text(body.get("system", "")):
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role, content = msg["role"], msg.get("content", "")
        is_asst = role == "assistant"

        # Simple string content
        if isinstance(content, str):
            out = {"role": role, "content": content}
            if thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
            continue

        if not isinstance(content, list):
            continue

        text_parts, tool_calls, thinking_text, tool_results = [], [], "", []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue

            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_text = block.get("thinking", "")
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _extract_text(block.get("content", "")),
                })

        # Emit tool_result messages first (must immediately follow assistant's tool_calls)
        messages.extend(tool_results)

        # Then emit the main message (text + tool_calls + thinking)
        if tool_calls:
            out = {
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "",
                "tool_calls": tool_calls,
            }
            if thinking_text:
                out["reasoning_content"] = thinking_text
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
        elif text_parts or thinking_text or (thinking and is_asst):
            out = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
            if thinking_text:
                out["reasoning_content"] = thinking_text
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)

    # Build request
    oai = {"model": model, "messages": messages,
           "max_tokens": body.get("max_tokens", 8096),
           "stream": body.get("stream", False)}

    for key, oai_key in [("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop")]:
        if key in body:
            oai[oai_key] = body[key]

    if "tools" in body:
        oai["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }} for t in body["tools"]]
        tc = body.get("tool_choice", "auto")
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "tool":
                oai["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
            elif tc_type == "any":
                oai["tool_choice"] = "required"
            else:
                oai["tool_choice"] = "auto"
        else:
            oai["tool_choice"] = tc

    return oai


def openai_to_anthropic(resp: dict, model: str) -> dict:
    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = resp.get("usage", {})

    blocks = []
    if reasoning := msg.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": reasoning})
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except Exception:
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""), "input": inp})

    stop = "tool_use" if msg.get("tool_calls") else "end_turn"
    if choice.get("finish_reason") == "length":
        stop = "max_tokens"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "content": blocks, "model": model, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)},
    }


def _build_display() -> Group:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False, expand=False)
    table.add_column("Route", style="bold", width=8)
    table.add_column("Model", style="bold", min_width=14)
    table.add_column("Total", justify="right", min_width=10)
    table.add_column("Input", justify="right", min_width=10)
    table.add_column("Output", justify="right", min_width=10)
    table.add_column("Cache", justify="right", min_width=10)
    table.add_column("%", justify="right", min_width=6)

    # Compute grand total first for percentage calculation
    with _token_lock:
        usage_snapshot = {m: dict(d) for m, d in _token_usage.items()}

    sum_total = 0
    for route_info in ROUTES.values():
        d = usage_snapshot[route_info["model"]]
        sum_total += d["input"] + d["output"] + d["cache"]

    sum_in = sum_out = sum_cache = 0
    shown = set()
    for route_name, route_info in ROUTES.items():
        model = route_info["model"]
        shown.add(model)
        d = usage_snapshot[model]
        total = d["input"] + d["output"] + d["cache"]
        sum_in += d["input"]
        sum_out += d["output"]
        sum_cache += d["cache"]
        pct = f"{total / sum_total * 100:.1f}%" if sum_total else "0%"
        table.add_row(
            route_name, model,
            f"{total:,}", f"{d['input']:,}", f"{d['output']:,}", f"{d['cache']:,}",
            pct,
        )

    # Show any other models with usage not covered by routes
    for model, d in usage_snapshot.items():
        if model in shown:
            continue
        total = d["input"] + d["output"] + d["cache"]
        if total == 0:
            continue
        sum_in += d["input"]
        sum_out += d["output"]
        sum_cache += d["cache"]
        pct = f"{total / sum_total * 100:.1f}%" if sum_total else "0%"
        table.add_row(
            "-", model,
            f"{total:,}", f"{d['input']:,}", f"{d['output']:,}", f"{d['cache']:,}",
            pct,
        )

    sum_total = sum_in + sum_out + sum_cache
    table.add_row(
        "[bold yellow]ALL[/]", "",
        f"[bold yellow]{sum_total:,}[/]",
        f"[bold yellow]{sum_in:,}[/]",
        f"[bold yellow]{sum_out:,}[/]",
        f"[bold yellow]{sum_cache:,}[/]",
        f"[bold yellow]100%[/]",
    )

    start = max(0, min(_log_scroll, len(_log_lines) - _LOG_VISIBLE))
    visible = list(_log_lines)[start:start + _LOG_VISIBLE]
    log_text = "\n".join(visible) if visible else "[dim]waiting for requests...[/]"
    if len(_log_lines) > _LOG_VISIBLE:
        log_text += f"\n[dim]↑ {start + 1}/{len(_log_lines)} logs (scroll with ↑↓ keys)[/]"

    return Group(
        Panel(table, title="[bold green]Token Usage[/]", border_style="green", padding=(0, 1)),
        Panel(log_text, title="[bold]Log[/]", border_style="dim", padding=(0, 1)),
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _estimate_input_tokens(body: dict) -> int:
    """Estimate input tokens from message content, tools, and tool_results."""
    total = 0

    # System prompt
    system = body.get("system", "")
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        for s in system:
            if isinstance(s, str):
                total += len(s)
            elif isinstance(s, dict):
                total += len(s.get("text", ""))

    # Tools definitions
    for tool in body.get("tools", []):
        total += len(tool.get("name", ""))
        total += len(tool.get("description", ""))
        total += len(str(tool.get("input_schema", {})))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    total += len(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        total += len(_extract_text(block.get("content", "")))
                    elif btype == "thinking":
                        total += len(block.get("thinking", ""))
                    else:
                        total += len(block.get("text", "")) + len(str(block.get("input", "")))

    return max(1, total // 3)


def _elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)


@app.api_route("/v1/messages", methods=["POST"])
@app.api_route("/anthropic/v1/messages", methods=["POST"])
async def messages(request: Request):
    req_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.time()

    try:
        body = json.loads(await request.body())
    except Exception:
        return Response(content='{"error":"invalid json"}', status_code=400)

    original_model = body.get("model", "")
    route = _route_for(original_model)
    model_id = route["model"]
    cfg = get_model_config(model_id)
    endpoint = cfg["endpoint"]
    protocol = cfg["protocol"]

    body = dict(body)
    body["model"] = model_id

    # Extract thinking for logging
    thinking = body.get("thinking", {})
    thinking_type = thinking.get("type", "none") if isinstance(thinking, dict) else "none"
    effort = (body.get("effort")
              or (thinking.get("effort") if isinstance(thinking, dict) else None)
              or (body.get("output_config", {}).get("effort") if isinstance(body.get("output_config"), dict) else None)
              or "none")

    _log(f"→ {original_model!r} → {model_id} | {protocol} | stream={body.get('stream', False)} | thinking={thinking_type} | effort={effort}")

    # ── Anthropic pass-through ──────────────────────────────────
    if protocol == "anthropic":
        a_headers = {"x-api-key": API_KEY, "Content-Type": "application/json",
                     "anthropic-version": "2023-06-01"}
        is_stream = body.get("stream", False)

        if not is_stream:
            resp = await _client.post(endpoint, json=body, headers=a_headers)
            if resp.status_code != 200:
                _log(f"  ERROR {resp.status_code}: {resp.text[:300]}")
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
            # Track token usage (Anthropic or OpenAI format)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            usage = data.get("usage", {})
            req_in = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            req_out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            req_cache = usage.get("cache_read_input_tokens", 0)
            with _token_lock:
                _token_usage[model_id]["input"] += req_in
                _token_usage[model_id]["output"] += req_out
                _token_usage[model_id]["cache"] += req_cache
            _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{req_cache} cache")
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         req_in, req_out, req_cache, success=True)
            return Response(content=resp.content, media_type="application/json")

        # Estimate input tokens for Anthropic streaming (MiniMax may not send usage in message_start)
        est_input = _estimate_input_tokens(body)
        with _token_lock:
            _token_usage[model_id]["input"] += est_input

        async def anthropic_stream():
            stream_in = stream_out = stream_cache = 0
            try:
                async with _client.stream("POST", endpoint, json=body, headers=a_headers) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        _log(f"  ERROR {resp.status_code}: {err[:300]}")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
                        error_payload = {"type": "error", "error": {"type": "api_error",
                                       "message": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='replace')[:200]}"}}
                        yield _sse("error", error_payload)
                        return
                    async for chunk in resp.aiter_bytes():
                        # Parse SSE chunks to track token usage
                        for line in chunk.decode("utf-8", errors="replace").split("\n"):
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except Exception:
                                continue
                            etype = event.get("type", "")
                            if etype == "message_start":
                                usage = event.get("message", {}).get("usage", {})
                                stream_in = usage.get("input_tokens")
                                if stream_in is not None:
                                    # Overwrite the estimate with actual value from API
                                    with _token_lock:
                                        _token_usage[model_id]["input"] -= est_input
                                        _token_usage[model_id]["input"] += stream_in
                            elif etype == "message_delta":
                                usage = event.get("usage", {})
                                stream_out = usage.get("output_tokens", 0)
                                with _token_lock:
                                    _token_usage[model_id]["output"] += stream_out
                        yield chunk
            except Exception as e:
                _log(f"  ERROR stream: {e}")
                if not stream_in:
                    with _token_lock:
                        _token_usage[model_id]["input"] -= est_input
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             stream_in or est_input, stream_out, stream_cache, success=False, error=str(e))
                return
            # Use estimated input if API didn't provide usage in message_start
            logged_in = stream_in or est_input
            if stream_in or stream_out:
                _log(f"  ← {model_id} | +{logged_in} in (est) | +{stream_out} out | +{stream_cache} cache")
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             logged_in, stream_out, stream_cache, success=True)

        return StreamingResponse(anthropic_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    # ── OpenAI-protocol (existing logic) ────────────────────────
    oai_body = anthropic_to_openai(body, model_id)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    is_stream = oai_body["stream"]

    if not is_stream:
        resp = await _client.post(endpoint, json=oai_body, headers=headers)
        if resp.status_code != 200:
            _log(f"  ERROR {resp.status_code}: {resp.text[:300]}")
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        data = resp.json()
        usage = data.get("usage", {})
        req_in = usage.get("prompt_tokens", 0)
        req_out = usage.get("completion_tokens", 0)
        cache = (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or usage.get("cached_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
        )
        with _token_lock:
            _token_usage[model_id]["input"] += req_in
            _token_usage[model_id]["output"] += req_out
            _token_usage[model_id]["cache"] += cache
        _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{cache} cache")
        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                     req_in, req_out, cache, success=True)
        return Response(content=json.dumps(openai_to_anthropic(data, original_model), ensure_ascii=False),
                        media_type="application/json")

    # Streaming
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Request usage in stream for accurate tracking
    oai_body["stream_options"] = {"include_usage": True}

    # Estimate input tokens from message content
    stream_in_est = _estimate_input_tokens(body)
    with _token_lock:
        _token_usage[model_id]["input"] += stream_in_est

    async def stream_gen():
        started = False
        open_blocks = []  # Ordered list of open block indices
        text_block_idx = None
        reasoning_block_idx = None
        tool_block_idx = {}  # api_idx -> block_idx
        next_block_idx = 0
        stream_out_tokens = 0
        actual_usage = None

        try:
            async with _client.stream("POST", endpoint, json=oai_body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    _log(f"  ERROR {resp.status_code}: {err[:300]}")
                    _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                 0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
                    error_payload = {"type": "error", "error": {"type": "api_error",
                                   "message": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='replace')[:200]}"}}
                    yield _sse("error", error_payload)
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()

                    if data == "[DONE]":
                        # Use actual usage from stream if available, otherwise use estimates
                        final_in = stream_in_est
                        final_out = stream_out_tokens
                        final_cache = 0
                        with _token_lock:
                            if actual_usage:
                                final_in = actual_usage.get("prompt_tokens")
                                if final_in is None:
                                    final_in = stream_in_est
                                final_out = actual_usage.get("completion_tokens")
                                if final_out is None:
                                    final_out = stream_out_tokens
                                final_cache = (
                                    actual_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                    or actual_usage.get("cached_tokens", 0)
                                    or actual_usage.get("cache_read_input_tokens", 0)
                                )
                                _token_usage[model_id]["input"] -= stream_in_est
                                _token_usage[model_id]["input"] += final_in
                                _token_usage[model_id]["output"] += final_out
                                if final_cache:
                                    _token_usage[model_id]["cache"] += final_cache
                            else:
                                # No actual usage: add the estimate we accumulated during streaming
                                _token_usage[model_id]["output"] += stream_out_tokens
                        for idx in open_blocks:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                        has_tools = bool(tool_block_idx)
                        yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use" if has_tools else "end_turn"}, "usage": {"output_tokens": 0}})
                        yield _sse("message_stop", {"type": "message_stop"})
                        log_tag = "" if actual_usage else " (est)"
                        _log(f"  ← {model_id} | +{final_in} in{log_tag} | +{final_out} out{log_tag} | +{final_cache} cache")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     final_in, final_out, final_cache, success=True)
                        break

                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue

                    # Capture usage from stream (sent in final chunk when stream_options.include_usage=true)
                    chunk_usage = chunk.get("usage")
                    if chunk_usage and isinstance(chunk_usage, dict):
                        actual_usage = chunk_usage

                    choices = chunk.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        continue
                    first_choice = choices[0] if choices else {}
                    delta = first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
                    if not delta or not isinstance(delta, dict):
                        delta = {}

                    if not started:
                        started = True
                        yield _sse("message_start", {"type": "message_start", "message": {
                            "id": msg_id, "type": "message", "role": "assistant", "content": [],
                            "model": original_model, "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}}})

                    # Text
                    text = ""
                    c = delta.get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")

                    if text:
                        if text_block_idx is None:
                            text_block_idx = next_block_idx
                            next_block_idx += 1
                            yield _sse("content_block_start", {"type": "content_block_start", "index": text_block_idx,
                                       "content_block": {"type": "text", "text": ""}})
                            open_blocks.append(text_block_idx)
                        stream_out_tokens += _estimate_tokens(text)
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx,
                                   "delta": {"type": "text_delta", "text": text}})

                    # Reasoning content
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        if reasoning_block_idx is None:
                            reasoning_block_idx = next_block_idx
                            next_block_idx += 1
                            yield _sse("content_block_start", {"type": "content_block_start", "index": reasoning_block_idx,
                                       "content_block": {"type": "thinking", "thinking": ""}})
                            open_blocks.append(reasoning_block_idx)
                        stream_out_tokens += _estimate_tokens(reasoning)
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": reasoning_block_idx,
                                   "delta": {"type": "thinking_delta", "thinking": reasoning}})

                    # Tool calls
                    for tc in (delta.get("tool_calls") or []):
                        api_idx = tc.get("index", 0)
                        if api_idx not in tool_block_idx:
                            block_idx = next_block_idx
                            next_block_idx += 1
                            tool_block_idx[api_idx] = block_idx
                            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}")
                            yield _sse("content_block_start", {"type": "content_block_start", "index": block_idx,
                                       "content_block": {"type": "tool_use", "id": tc_id,
                                       "name": tc.get("function", {}).get("name", ""), "input": {}}})
                            open_blocks.append(block_idx)
                        if args := tc.get("function", {}).get("arguments", ""):
                            stream_out_tokens += _estimate_tokens(args)
                            yield _sse("content_block_delta", {"type": "content_block_delta", "index": tool_block_idx[api_idx],
                                       "delta": {"type": "input_json_delta", "partial_json": args}})
        except Exception as e:
            _log(f"  ERROR stream: {e}")
            with _token_lock:
                _token_usage[model_id]["input"] -= stream_in_est
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         stream_in_est, stream_out_tokens, 0, success=False, error=str(e))
            if started:
                for idx in open_blocks:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "error"}, "usage": {"output_tokens": 0}})
                yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(stream_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/health")
async def health():
    with _db_lock:
        _conn.execute("SELECT 1")
    with _token_lock:
        usage = {model: {"input": d["input"], "output": d["output"], "cache": d["cache"]}
                 for model, d in _token_usage.items()}
    return {"status": "ok", "usage": usage}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Đếm tokens cho Anthropic messages request."""
    try:
        body = json.loads(await request.body())
    except Exception:
        return Response(content='{"error":"invalid json"}', status_code=400)

    # Sử dụng hàm _estimate_input_tokens đã có
    tokens = _estimate_input_tokens(body)
    return {"input_tokens": tokens}


@app.get("/api/stats")
async def get_stats(from_date: str = None, to_date: str = None):
    where, params = _build_where(from_date, to_date)

    with _db_lock:
        row = _conn.execute(
            "SELECT COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),"
            "       COALESCE(SUM(tokens_cache), 0), COUNT(*),"
            "       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),"
            "       SUM(CASE WHEN success = 0 OR success IS NULL THEN 1 ELSE 0 END),"
            "       COALESCE(AVG(duration_ms), 0)"
            " FROM requests " + where,
            params,
        ).fetchone()

        totals = {
            "input": row[0], "output": row[1], "cache": row[2],
            "total": row[0] + row[1] + row[2],
            "count": row[3],
            "success_count": row[4], "fail_count": row[5],
            "avg_duration_ms": int(row[6])
        }

        rows = _conn.execute(
            "SELECT model, COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),"
            "       COALESCE(SUM(tokens_cache), 0), COUNT(*),"
            "       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),"
            "       SUM(CASE WHEN success = 0 OR success IS NULL THEN 1 ELSE 0 END),"
            "       COALESCE(AVG(duration_ms), 0)"
            " FROM requests " + where +
            " GROUP BY model",
            params,
        ).fetchall()

    sum_total = totals["total"]
    models = {}
    for r in rows:
        t = r[1] + r[2] + r[3]
        models[r[0]] = {
            "input": r[1], "output": r[2], "cache": r[3], "total": t,
            "pct": f"{t/sum_total*100:.1f}%" if sum_total else "0%",
            "count": r[4], "success_count": r[5], "fail_count": r[6],
            "avg_duration_ms": int(r[7])
        }

    return {"models": models, "totals": totals}


@app.get("/api/logs")
async def get_logs(limit: int = 100, offset: int = 0):
    lines = list(_log_lines)
    return {
        "logs": lines[offset:offset+limit],
        "total": len(lines),
        "has_more": offset + limit < len(lines)
    }


def _build_where(from_date: str = None, to_date: str = None):
    conditions, params = [], []
    if from_date:
        conditions.append("timestamp >= ?")
        params.append(from_date)
    if to_date:
        conditions.append("timestamp <= ?")
        params.append(to_date + "T23:59:59")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


@app.get("/api/history")
async def get_history(from_date: str = None, to_date: str = None, limit: int = 20, offset: int = 0):
    where, params = _build_where(from_date, to_date)
    query = "SELECT * FROM requests " + where + " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _db_lock:
        rows = _conn.execute(query, params).fetchall()
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM requests " + where
        total_row = _conn.execute(count_query, params[:-2]).fetchone()
        total_count = total_row[0] if total_row else 0

    return {
        "logs": [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "model": r["model"],
                "original_model": r["original_model"],
                "duration_ms": r["duration_ms"],
                "tokens_input": r["tokens_input"],
                "tokens_output": r["tokens_output"],
                "tokens_cache": r["tokens_cache"],
                "success": bool(r["success"]),
                "error": r["error"]
            }
            for r in rows
        ],
        "total": total_count,
        "page": offset // limit + 1,
        "per_page": limit,
        "has_more": offset + limit < total_count
    }


@app.delete("/api/history")
async def delete_history(before: str = None, all: bool = False):
    with _db_lock:
        if all:
            _conn.execute("DELETE FROM requests")
        elif before:
            _conn.execute("DELETE FROM requests WHERE timestamp < ?", (before + "T23:59:59",))
        else:
            return {"error": "Specify 'before' date or 'all=true'"}
        _conn.commit()
    return {"status": "deleted"}


if __name__ == "__main__":
    import threading as th
    from uvicorn import Config, Server

    # Redirect uvicorn logs to Rich buffer
    h = _RichLogHandler()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [h]
        lg.propagate = False

    # Main API server
    config = Config(app, host=HOST, port=PORT, log_level="info", log_config=None)
    server = Server(config)

    thread = th.Thread(target=server.run, daemon=True)
    thread.start()

    # Start web UI on separate port (if different from PORT)
    if WEB_PORT != PORT:
        web_config = Config(app, host=HOST, port=WEB_PORT, log_level="info", log_config=None)
        web_server = Server(web_config)
        web_thread = th.Thread(target=web_server.run, daemon=True)
        web_thread.start()

    time.sleep(0.5)  # Wait for servers to start
    _log(f"🔌 API: http://localhost:{PORT}")
    if WEB_PORT != PORT:
        _log(f"🌐 Web UI: http://localhost:{WEB_PORT}")

    _running = True

    def _input_thread():
        global _log_scroll, _running
        if sys.platform == "win32":
            import msvcrt
            while _running:
                if msvcrt.kbhit():
                    try:
                        ch = msvcrt.getch()
                        # Arrow keys on Windows: prefix b'\xe0' or b'\x00' + scan code
                        if ch in (b'\xe0', b'\x00'):
                            ch2 = msvcrt.getch()
                            if ch2 == b'H':       # Up arrow
                                _log_scroll = max(0, _log_scroll - 1)
                            elif ch2 == b'P':     # Down arrow
                                _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + 1)
                            elif ch2 == b'I':     # Page Up
                                _log_scroll = max(0, _log_scroll - _LOG_VISIBLE)
                            elif ch2 == b'Q':     # Page Down
                                _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + _LOG_VISIBLE)
                            elif ch2 == b'G':     # Home
                                _log_scroll = 0
                            elif ch2 == b'O':     # End
                                _log_scroll = max(0, len(_log_lines) - _LOG_VISIBLE)
                            continue
                        ch = ch.decode("utf-8", errors="ignore")
                        if ch == "\x03":          # Ctrl-C
                            _running = False
                        elif ch == "k":
                            _log_scroll = max(0, _log_scroll - 1)
                        elif ch == "j":
                            _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + 1)
                        elif ch == "G":
                            _log_scroll = max(0, len(_log_lines) - _LOG_VISIBLE)
                        elif ch == "g":
                            _log_scroll = 0
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)
        else:
            import tty, termios, select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setraw(fd)
            try:
                while _running:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch == "\x03":
                            _running = False
                        elif ch == "\x1b":       # Escape sequence (arrow keys)
                            seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                            if seq == "[A":       # Up
                                _log_scroll = max(0, _log_scroll - 1)
                            elif seq == "[B":     # Down
                                _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + 1)
                            elif seq == "[5":     # Page Up
                                if sys.stdin.read(1) == "~":
                                    _log_scroll = max(0, _log_scroll - _LOG_VISIBLE)
                            elif seq == "[6":     # Page Down
                                if sys.stdin.read(1) == "~":
                                    _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + _LOG_VISIBLE)
                            elif seq == "[H":     # Home
                                _log_scroll = 0
                            elif seq == "[F":     # End
                                _log_scroll = max(0, len(_log_lines) - _LOG_VISIBLE)
                        elif ch in ("k",):
                            _log_scroll = max(0, _log_scroll - 1)
                        elif ch in ("j",):
                            _log_scroll = min(max(0, len(_log_lines) - _LOG_VISIBLE), _log_scroll + 1)
                        elif ch == "G":
                            _log_scroll = max(0, len(_log_lines) - _LOG_VISIBLE)
                        elif ch == "g":
                            _log_scroll = 0
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    th.Thread(target=_input_thread, daemon=True).start()

    with Live(_build_display(), refresh_per_second=1, screen=True) as live:
        while _running:
            live.update(_build_display())
            time.sleep(1)