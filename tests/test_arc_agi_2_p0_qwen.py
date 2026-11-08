"""Tests for the Qwen3.8 P0 probe (runs/arc-agi-2/p0-qwen/src): pure helpers and the request/metrics loop against a
mock OpenAI-compatible SSE server. Run: .venv/bin/python -m unittest tests.test_arc_agi_2_p0_qwen -v"""
import importlib
import json
import os
import random
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "runs/arc-agi-2/p0-qwen/src"
sys.path.insert(0, str(SRC))
import arc_llm as L  # noqa: E402


def rand_grid(rng):
    h, w = rng.randint(1, 7), rng.randint(1, 7)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


class Helpers(unittest.TestCase):
    def test_dihedral_roundtrip_and_distinct(self):
        rng = random.Random(0)
        for _ in range(200):
            g, k = rand_grid(rng), rng.randrange(8)
            self.assertEqual(L.dihedral_inverse(L.dihedral(g, k), k), g)
        g = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(len({json.dumps(L.dihedral(g, k)) for k in range(8)}), 8)

    def test_aug_roundtrip_nonidentity_and_zero_fixed(self):
        rng = random.Random(1)
        for tid in ["dbff022c", "abc82100", "x", "y" * 9]:
            aug = L.augmentation(tid)
            self.assertIn(aug["k"], range(1, 8))
            self.assertEqual(aug["perm"][0], 0)
            self.assertEqual(sorted(aug["perm"]), list(range(10)))
            self.assertEqual(aug, L.augmentation(tid))  # deterministic
            for _ in range(20):
                g = rand_grid(rng)
                self.assertEqual(L.invert_aug(L.apply_aug(g, aug), aug), g)

    def test_parse_answer(self):
        self.assertEqual(L.parse_answer("blah\n```json\n[[1,2],[3,4]]\n```"), [[1, 2], [3, 4]])
        self.assertEqual(L.parse_answer("```json\n[[1]]\n```\nthen ```json\n[[2,2]]\n```"), [[2, 2]])
        self.assertEqual(L.parse_answer("answer: [[0, 1], [1, 0]]"), [[0, 1], [1, 0]])
        self.assertIsNone(L.parse_answer("```json\n[[1,2],[3]]\n```"))  # ragged
        self.assertIsNone(L.parse_answer("[[10]]"))
        self.assertIsNone(L.parse_answer(""))
        self.assertIsNone(L.parse_answer("[[1,2]"))
        self.assertEqual(L.parse_answer("x [[3]] y ```\n[[4]]\n```"), [[4]])

    def test_prompt_contains_every_grid_and_only_one_test(self):
        task = {"train": [{"input": [[1]], "output": [[2]]}, {"input": [[3, 3]], "output": [[4, 4]]}],
                "test": [{"input": [[5]]}, {"input": [[6, 6, 6]]}]}
        p = L.build_prompt(task, 1)
        for s in ("[[1]]", "[[2]]", "[[3,3]]", "[[4,4]]", "[[6,6,6]]"):
            self.assertIn(s, p)
        self.assertNotIn("[[5]]", p)

    def test_summaries(self):
        rows = [{"key": "a:0", "state": "done", "completion_tokens": 100, "finish_reason": "stop", "answer": [[1]],
                 "correct": True},
                {"key": "a:0", "state": "done", "completion_tokens": 300, "finish_reason": "stop", "answer": [[1]],
                 "correct": True},
                {"key": "b:0", "state": "done", "completion_tokens": 500, "finish_reason": "length", "answer": None,
                 "correct": False},
                {"key": "b:0", "state": "aborted", "completion_tokens": 50},
                {"key": "c:0", "state": "not_started"}]
        s = L.arm_summary(rows)
        self.assertEqual((s["finished"], s["truncated"], s["parsed"], s["correct"]), (3, 1, 2, 2))
        self.assertEqual((s["tokens"]["n_done"], s["tokens"]["n_censored"], s["tokens"]["sum_all"]), (3, 1, 950))
        self.assertEqual(s["tokens"]["p50"], 300)
        a = L.pair_agreement(rows)
        self.assertEqual((a["pairs"], a["agree"], a["agree_correct"]), (1, 1, 1))

    def test_km_quantiles_handle_censoring(self):
        # no censoring: KM median equals the empirical lower median
        self.assertEqual(L.km_quantiles([1, 2, 3, 4], [])["km_p50"], 2)
        # heavy censoring above 2: median not reached among events -> pushed up or None, never below the naive one
        km = L.km_quantiles([1, 2], [5, 5, 5, 5])
        self.assertIsNone(km["km_p50"])
        km2 = L.km_quantiles([1, 2, 6, 7], [3, 3])
        self.assertGreaterEqual(km2["km_p50"], L.quantiles([1, 2, 6, 7])["p50"])
        self.assertEqual(L.token_stats([{"state": "aborted", "completion_tokens": 90000}])["censored_over_80k"], 1)


# --- mock vLLM ---------------------------------------------------------------------------------------------------------

class Mock(BaseHTTPRequestHandler):
    gen_total = 0.0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "q38"}]})
        if self.path == "/metrics":
            text = (f"# HELP x\nvllm:generation_tokens_total{{model_name=\"q38\"}} {Mock.gen_total}\n"
                    "vllm:num_requests_running{model_name=\"q38\"} 2\nvllm:kv_cache_usage_perc 0.5\n"
                    "vllm:spec_decode_num_accepted_tokens_total 30\nvllm:spec_decode_num_draft_tokens_total 60\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            return self.wfile.write(text)
        self._json({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json({"count": 500})
        prompt = body["messages"][0]["content"]
        # The mock "solves" the task by reading the solution the test planted in the prompt's last line.
        answer = prompt.split("PLANTED:")[-1].strip() if "PLANTED:" in prompt else "[[9]]"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        n = 0
        for i in range(5):
            n += 10
            ev = {"choices": [{"delta": {"reasoning_content": "thinking " * 3}}],
                  "usage": {"prompt_tokens": 500, "completion_tokens": n}}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        trunc = body["chat_template_kwargs"]["reasoning_effort"] == "low" and "TRUNC" in prompt
        last = {"choices": [{"delta": {"content": "" if trunc else f"Done.\n```json\n{answer}\n```"},
                             "finish_reason": "length" if trunc else "stop"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": n + 5}}
        self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
        with Mock.lock:
            Mock.gen_total += n + 5


class ProbeLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ.update(P0_OUT=cls.tmp.name, P0_WALL_SECONDS="3600", P0_TASKS="3")
        sys.modules.pop("p0_probe", None)
        cls.P = importlib.import_module("p0_probe")
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.tmp.cleanup()

    def _comp(self, keys):
        comp = Path(self.tmp.name) / "comp"
        comp.mkdir(exist_ok=True)
        ch, so = {}, {}
        for n, k in enumerate(keys):
            sol = [[n % 10, 1], [2, 3]]
            ch[k] = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6]]}]}
            so[k] = [sol]
        (comp / "arc-agi_evaluation_challenges.json").write_text(json.dumps(ch))
        (comp / "arc-agi_evaluation_solutions.json").write_text(json.dumps(so))
        return comp, so

    def test_request_inverts_augmentation_and_scores(self):
        P = self.P
        keys = P.KEYS30[:3]
        comp, so = self._comp(keys)
        _, jobs = P.build_jobs(comp)
        self.assertEqual(len(jobs), 9)  # 3 tasks x 1 output x (xhigh, low, low)
        self.assertEqual(len({j["seed"] for j in jobs}), 9)
        job = dict(jobs[0])
        planted = L.apply_aug(job["solution"], job["aug"])  # what a correct model answers in augmented space
        job["prompt"] += "\nPLANTED:" + json.dumps(planted)
        row = P.one_request(job, self.port, time.time() + 60, stop_at=time.time() + 60)
        self.assertEqual(row["state"], "done")
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual(row["completion_tokens"], 55)
        self.assertEqual(row["answer"], job["solution"])
        self.assertTrue(row["correct"])
        wrong = P.one_request(dict(jobs[1]), self.port, time.time() + 60, stop_at=time.time() + 60)
        self.assertFalse(wrong["correct"])
        trunc = dict(jobs[2])
        trunc["prompt"] += "\nTRUNC"
        t = P.one_request(trunc, self.port, time.time() + 60, stop_at=time.time() + 60)
        self.assertEqual((t["state"], t["finish_reason"], t["answer"]), ("done", "length", None))
        late = P.one_request(dict(jobs[0]), self.port, time.time() + 60, stop_at=time.time() - 1)
        self.assertEqual(late["state"], "not_started")

    def test_p1_load_keeps_streams_busy_until_end(self):
        sys.modules.pop("p1_sweep", None)
        p1 = importlib.import_module("p1_sweep")
        p1.PORT = self.port
        comp, _ = self._comp(self.P.KEYS30[:3])
        _, jobs = self.P.build_jobs(comp)
        jobs = [j for j in jobs if j["effort"] == "xhigh"]
        t = time.time()
        p1.load({"name": "t", "conc": 3}, jobs, time.time() + 1.5)
        self.assertLess(time.time() - t, 10)
        rows = [json.loads(x) for x in (Path(self.tmp.name) / "requests.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(sum(1 for r in rows if r.get("phase") == "t"), 3)

    def test_phase_main_writes_summary_and_metrics(self):
        P = self.P
        keys = P.KEYS30[:3]
        comp, _ = self._comp(keys)
        P.phase_main(comp, {"max_concurrency_at_len": 2.0}, port=self.port)
        main = P.S["phases"]["main"]
        self.assertEqual(main["jobs"], 9)
        self.assertEqual(main["arms"]["xhigh"]["requests"], 3)
        self.assertEqual(main["arms"]["low"]["finished"], 6)
        rows = (Path(self.tmp.name) / "requests.jsonl").read_text().splitlines()
        self.assertGreaterEqual(len(rows), 9)
        self.assertTrue((Path(self.tmp.name) / "summary.json").exists())
        v = P.scrape(self.port)
        self.assertIn("vllm:generation_tokens_total", v)
        self.assertEqual(v["vllm:num_requests_running"], 2)


if __name__ == "__main__":
    unittest.main()


class ProgSynth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SRC.parents[1] / "p0b-colab/src"))
        cls.P = importlib.import_module("progsynth")

    def test_extract_and_verify_and_feedback(self):
        P = self.P
        task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                          {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}],
                "test": [{"input": [[8, 9]]}]}
        good = "thinking...\n```python\ndef transform(grid):\n    return [row[::-1] for row in grid]\n```"
        code = P.extract_code(good)
        self.assertIn("def transform", code)
        chk = P.check(code, task)
        self.assertTrue(chk["verified"])
        self.assertEqual(chk["tests"], [[[9, 8]]])
        bad = P.check("def transform(grid):\n    return grid\n", task)
        self.assertFalse(bad["verified"])
        self.assertEqual(bad["train_pass"], 0)
        fb = P.feedback(task, bad)
        self.assertIn("0 of 2", fb)
        self.assertIn("expected", fb)
        crash = P.check("def transform(grid):\n    raise ValueError('x')\n", task)
        self.assertFalse(crash["verified"])
        self.assertIn("error", P.feedback(task, crash))
        self.assertFalse(P.check("def nope(:\n", task)["verified"])
        self.assertIsNone(P.extract_code("```python\nprint(1)\n```"))
        loop = P.run_program("def transform(grid):\n    while True: pass\n", [[[1]]], timeout=5)
        self.assertIsNone(loop[0])

    def test_prompt_has_tests_and_examples(self):
        task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3, 3]]}, {"input": [[4]]}]}
        p = self.P.build_prompt(task)
        for s in ("[[1]]", "[[2]]", "[[3,3]]", "[[4]]", "transform(grid)"):
            self.assertIn(s, p)
