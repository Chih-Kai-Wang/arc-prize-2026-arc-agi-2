"""Stub OpenAI-compatible server for local v13 tests (stdlib only). Behaviour per request, chosen by the request seed:
seed % 4 == 0: identity program in the final answer (stop); 1: truncated (length) with an identity draft only in the
reasoning; 2: a wrong program (stop); 3: no code at all (stop). /tokenize counts characters/3 (messages) or returns
one id per character (prompt). v3 budget forcing: return_token_ids streams prompt_token_ids + per-chunk token_ids;
FAKE_FORCE=1 makes mode 1 a truncation with no draft, and /v1/completions (token-id prompt that must be the chat
prompt ids + the generated ids + the forced suffix) writes the identity body; FAKE_FORCE=cut stops it inside the block;
FAKE_FORCE=answer makes mode 1 leave its thinking (the "</think>" id among the output ids) and stop at the length limit
in a code-less answer, continued without a suffix. "</think>" tokenizes to one special id. FAKE_COMPL400=jobs rejects
job continuations (max_tokens > 32; the server-start parity probe still passes), =all rejects every continuation."""
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
            if "prompt" in body:
                toks = tokenize(body["prompt"])
                return self._json(200, {"count": len(toks), "tokens": toks})
            n = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 3
            return self._json(200, {"count": n, "tokens": []})
        if self.path == "/v1/completions":
            return self._completion(body)
        if self.path != "/v1/chat/completions":
            return self._json(404, {})
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})
        mode = body.get("seed", 0) % 4
        if os.environ.get("FAKE_HTTP500") and mode == 3:
            return self._json(500, {"error": "boom"})
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
        force = os.environ.get("FAKE_FORCE")
        reasoning = "Let me think. " + (f"Draft:\n{IDENTITY}\nhmm" if mode == 1 and not force else "maybe")
        content = {0: "Answer:\n" + IDENTITY, 1: "", 2: WRONG, 3: "I am not sure."}[mode]
        if mode == 1 and force == "answer":
            content = "The rule is clear; I will now write the"
        finish = "length" if mode == 1 else "stop"
        self._sse_start()
        n = 0
        ids = body.get("return_token_ids")
        if ids:
            self._chunk(f"data: {json.dumps({'prompt_token_ids': PROMPT_IDS, 'choices': []})}\n\n")
        pieces = [{"reasoning_content": reasoning[i:i + 20]} for i in range(0, len(reasoning), 20)]
        n_reason = len(pieces)
        pieces += [{"content": content[i:i + 20]} for i in range(0, len(content), 20)]
        for k, piece in enumerate(pieces):
            ev = {"choices": [{"delta": piece, "finish_reason": None}], "usage": {"completion_tokens": n + 5}}
            if ids:
                ev["choices"][0]["token_ids"] = list(range(1000 + n, 1005 + n - bool(os.environ.get("FAKE_BAD_IDS"))))
                if k == n_reason:  # the first answer chunk carries the end-of-thinking id
                    ev["choices"][0]["token_ids"][0] = THINK
            n += 5
            self._chunk(f"data: {json.dumps(ev)}\n\n")
            time.sleep(0.005)
        ev = {"choices": [{"delta": {}, "finish_reason": finish}], "usage": {"completion_tokens": n, "prompt_tokens": 9}}
        self._chunk(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


    def _completion(self, body):
        p = body.get("prompt")
        bad = os.environ.get("FAKE_COMPL400")
        if bad == "all" or (bad == "jobs" and body.get("max_tokens", 0) > 32):
            return self._json(400, {"error": "rejected"})
        ok_ids = isinstance(p, list) and all(type(t) is int for t in p) and p[:len(PROMPT_IDS)] == PROMPT_IDS
        out = p[len(PROMPT_IDS):] if ok_ids else []
        if ok_ids and out[-len(SUFFIX_IDS):] == SUFFIX_IDS and THINK not in out[:-len(SUFFIX_IDS)]:
            n_out = len(out) - len(SUFFIX_IDS)  # forced: prompt ids + generated ids + suffix
            ok = n_out > 0 and out[:n_out] == list(range(1000, 1000 + n_out))
            text = BODY if os.environ.get("FAKE_FORCE") != "cut" else BODY.split("```")[0][:-1]
        else:  # already answering: prompt ids + generated ids (end-of-thinking id among them), no suffix
            ok = ok_ids and THINK in out and out[-len(SUFFIX_IDS):] != SUFFIX_IDS
            text = " program.\n```python\n" + BODY
        if not ok:
            return self._json(400, {"error": "bad continuation prompt"})
        finish = "stop" if os.environ.get("FAKE_FORCE") != "cut" else "length"
        self._sse_start()
        for i in range(0, len(text), 20):
            ev = {"choices": [{"text": text[i:i + 20], "finish_reason": None}], "usage": {"completion_tokens": i // 4}}
            self._chunk(f"data: {json.dumps(ev)}\n\n")
        ev = {"choices": [{"text": "", "finish_reason": finish}], "usage": {"completion_tokens": len(text) // 4}}
        self._chunk(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


PROMPT_IDS = [7, 8, 9]
THINK = 151668  # "</think>": one special id
FORCE_SUFFIX = "\n\nMy thinking budget is used up. I will commit to the most likely rule found so far and write the " \
               "complete program now.\n</think>\n\n```python\n"


def tokenize(text):
    toks = []
    for i, part in enumerate(text.split("</think>")):
        toks += ([THINK] if i else []) + [ord(c) for c in part]
    return toks


SUFFIX_IDS = tokenize(FORCE_SUFFIX)
BODY = "def transform(grid):\n    return [list(r) for r in grid]\n```\nDone."

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
