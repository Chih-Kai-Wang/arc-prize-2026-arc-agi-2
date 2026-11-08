"""Stub OpenAI-compatible server for local v13 tests (stdlib only). Behaviour per request, chosen by the request seed:
seed % 4 == 0: identity program in the final answer (stop); 1: truncated (length) with an identity draft only in the
reasoning; 2: a wrong program (stop); 3: no code at all (stop). /tokenize counts characters/3."""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IDENTITY = "```python\ndef transform(grid):\n    return [list(r) for r in grid]\n```"
WRONG = "```python\ndef transform(grid):\n    return [[0 for _ in r] for r in grid]\n```"


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
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})
        mode = body.get("seed", 0) % 4
        if os.environ.get("FAKE_SLOW"):  # never finishes: an identity draft in the reasoning, then endless thinking
            self._sse_start()
            try:
                first = {"choices": [{"delta": {"reasoning_content": f"Draft:\n{IDENTITY}\n"}, "finish_reason": None}]}
                self._chunk(f"data: {json.dumps(first)}\n\n")
                if os.environ["FAKE_SLOW"] == "stall":  # a preempted request: nothing more is ever sent
                    time.sleep(3600)
                while True:
                    ev = {"choices": [{"delta": {"reasoning_content": "still thinking "}, "finish_reason": None}]}
                    self._chunk(f"data: {json.dumps(ev)}\n\n")
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                return
        reasoning = "Let me think. " + (f"Draft:\n{IDENTITY}\nhmm" if mode == 1 else "maybe")
        content = {0: "Answer:\n" + IDENTITY, 1: "", 2: WRONG, 3: "I am not sure."}[mode]
        finish = "length" if mode == 1 else "stop"
        self._sse_start()
        n = 0
        for piece in [{"reasoning_content": reasoning[i:i + 20]} for i in range(0, len(reasoning), 20)] + \
                     [{"content": content[i:i + 20]} for i in range(0, len(content), 20)]:
            n += 5
            ev = {"choices": [{"delta": piece, "finish_reason": None}], "usage": {"completion_tokens": n}}
            self._chunk(f"data: {json.dumps(ev)}\n\n")
            time.sleep(0.005)
        ev = {"choices": [{"delta": {}, "finish_reason": finish}], "usage": {"completion_tokens": n, "prompt_tokens": 9}}
        self._chunk(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
