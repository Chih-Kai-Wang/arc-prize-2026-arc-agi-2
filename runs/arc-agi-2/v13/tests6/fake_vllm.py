"""Stub OpenAI-compatible server for local v13 tests (stdlib only). Behaviour per request, chosen by the request seed:
seed % 4 == 0: identity program in the final answer (stop); 1: truncated (length) with an identity draft only in the
reasoning; 2: a wrong program (stop); 3: no code at all (stop). /tokenize counts characters/3.
C2 seams: a repair request (the last user message is execution feedback) answers with IDENTITY; FAKE_REJECT_EFFORT
rejects a non-stream request with that reasoning_effort (HTTP 400); FAKE_LOG appends one JSON line per chat request.
The reply policy is `chat(body, closed)`; a test host where bind() is denied can drive it without a socket."""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IDENTITY = "```python\ndef transform(grid):\n    return [list(r) for r in grid]\n```"
WRONG = "```python\ndef transform(grid):\n    return [[0 for _ in r] for r in grid]\n```"
LOG_LOCK = threading.Lock()


def is_repair(body):
    msgs = body.get("messages") or []
    last = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
    return last.startswith("Your program reproduces") or last.startswith("Your code failed")


def log_chat(body, repair):
    path = os.environ.get("FAKE_LOG")
    if not path:
        return
    with LOG_LOCK, open(path, "a") as f:
        f.write(json.dumps({"max_tokens": body.get("max_tokens"), "stream": bool(body.get("stream")),
                            "reasoning_effort": (body.get("chat_template_kwargs") or {}).get("reasoning_effort"),
                            "seed": body.get("seed"), "repair": repair}) + "\n")


def blocks(reasoning, content, finish):
    evs, n = [], 0
    for piece in [{"reasoning_content": reasoning[i:i + 20]} for i in range(0, len(reasoning), 20)] + \
                 [{"content": content[i:i + 20]} for i in range(0, len(content), 20)]:
        n += 5
        evs.append({"choices": [{"delta": piece, "finish_reason": None}], "usage": {"completion_tokens": n}})
    evs.append({"choices": [{"delta": {}, "finish_reason": finish}],
                "usage": {"completion_tokens": n, "prompt_tokens": 9}})
    return [f"data: {json.dumps(ev)}\n\n" for ev in evs] + ["data: [DONE]\n\n"]


def slow_blocks(mode, closed):
    first = {"choices": [{"delta": {"reasoning_content": f"Draft:\n{IDENTITY}\n"}, "finish_reason": None}]}
    yield f"data: {json.dumps(first)}\n\n"
    if mode == "stall":  # a preempted request: nothing more is ever sent
        while not closed():
            time.sleep(0.1)
        return
    while not closed():
        ev = {"choices": [{"delta": {"reasoning_content": "still thinking "}, "finish_reason": None}]}
        yield f"data: {json.dumps(ev)}\n\n"
        time.sleep(0.2)


def chat(body, closed=lambda: False):
    """(status, kind, payload) for a chat request; kind "json" (object) or "sse" (text blocks)."""
    repair = is_repair(body)
    log_chat(body, repair)
    if not body.get("stream"):
        effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        if effort and effort == os.environ.get("FAKE_REJECT_EFFORT"):
            return 400, "json", {"error": "Unexpected reasoning effort"}
        return 200, "json", {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
    if repair:  # execution feedback: the identity program fixes the stub's wrong first answer
        return 200, "sse", blocks("Fix.\n", IDENTITY, "stop")
    mode = body.get("seed", 0) % 4
    if os.environ.get("FAKE_SLOW"):  # never finishes: an identity draft in the reasoning, then endless thinking
        return 200, "sse", slow_blocks(os.environ["FAKE_SLOW"], closed)
    reasoning = "Let me think. " + (f"Draft:\n{IDENTITY}\nhmm" if mode == 1 else "maybe")
    content = {0: "Answer:\n" + IDENTITY, 1: "", 2: WRONG, 3: "I am not sure."}[mode]
    return 200, "sse", blocks(reasoning, content, "length" if mode == 1 else "stop")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # streamed bodies use chunked transfer encoding, like uvicorn under vLLM

    def log_message(self, *a):
        pass

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, text):
        b = text.encode()
        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "m"}]})
        elif self.path == "/metrics":
            b = b"vllm:generation_tokens_total 123.0\nvllm:num_requests_running 2.0\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/tokenize":
            n = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 3
            return self._json(200, {"count": n, "tokens": []})
        if self.path != "/v1/chat/completions":
            return self._json(404, {})
        status, kind, payload = chat(body)
        if kind == "json":
            return self._json(status, payload)
        self._sse_start()
        try:
            for block in payload:
                self._chunk(block)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    port = ap.parse_args().port
    if os.environ.get("FAKE_NO_BIND"):  # test host without bind(): the client-side shim answers instead
        time.sleep(3600)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
