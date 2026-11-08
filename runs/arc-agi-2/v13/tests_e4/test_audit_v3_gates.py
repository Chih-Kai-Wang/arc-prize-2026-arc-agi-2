"""T3 review 10-16 audit fixes (zero GPU): v3 gates use the paired --exclude-source forced replay, block a lost correct
NVARC answer and net-negative forcing, require the pinned v2 baseline; the v3/v4 review gate needs two raw reviews
with matching sha256, no NO-GO and every P0/P1 refuted or fixed outside the kernel.
  .venv/bin/python runs/arc-agi-2/v13/tests_e4/test_audit_v3_gates.py"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import audit_v13 as A  # noqa: E402

LLM = {"force_at": 58000, "token_id_parity": {"ok": True}}
ROWS = ([{"task": "t1", "forced": True, "cont_state": "done", "cont_completion_tokens": 900, "verified": True,
          "source": "forced", "correct": [True]}] * 3
        + [{"task": "t2", "verified": True, "source": "content", "correct": [True]}])


def replay(d, name, after, delta, lost, excl=None):
    seeds = {str(s): {"after": a, "per_task_delta": delta} for s, a in zip((1, 2, 3), after)}
    p = Path(d) / name
    p.write_text(json.dumps({"seeds": seeds, "program_tasks": ["t1", "t2"], "rep": None, "exclude_source": excl,
                             "lost_correct_nvarc_any": lost}))
    return p


def test_paired_gates():
    with tempfile.TemporaryDirectory() as d:
        A.HERE  # keys30 read from src3: make t1/t2 part of it by patching the program-task check
        keys = set(json.loads((A.HERE / "src3/keys30.json").read_text()))
        k1, k2 = sorted(keys)[:2]
        rows = [dict(r, task={"t1": k1, "t2": k2}[r["task"]]) for r in ROWS]
        full = replay(d, "full.json", [9.5, 9.5, 9.5], {k1: 1.0, k2: 1.0}, False)
        excl = replay(d, "excl.json", [9.0, 9.0, 9.0], {k2: 1.0}, False, excl="forced")
        for p in (full, excl):
            j = json.loads(p.read_text()); j["program_tasks"] = sorted([k1, k2]); p.write_text(json.dumps(j))
        g, f = A.v3_gates(LLM, rows, full, excl)
        assert g["v2_baseline_pinned"] and g["v3_forced_contributed"] and g["v3_forcing_not_negative"], (g, f)
        assert g["v3_no_lost_correct_nvarc"] and g["v3_replay_mean_ge_v2"], (g, f)
        # forced answers displace a correct NVARC answer: forcing net-negative + lost correct -> gates fail
        bad = replay(d, "bad.json", [8.8, 8.8, 8.8], {k1: 0.0, k2: 1.0}, True)
        j = json.loads(bad.read_text()); j["program_tasks"] = sorted([k1, k2]); bad.write_text(json.dumps(j))
        g, f = A.v3_gates(LLM, rows, bad, excl)
        assert not g["v3_forced_contributed"] and not g["v3_forcing_not_negative"] and not g["v3_no_lost_correct_nvarc"], g
        # natural-stop gain on the forced task must not count as forcing's contribution (old bug)
        both = replay(d, "both.json", [9.0, 9.0, 9.0], {k1: 1.0, k2: 1.0}, False)
        exb = replay(d, "exb.json", [9.0, 9.0, 9.0], {k1: 1.0, k2: 1.0}, False, excl="forced")
        for p in (both, exb):
            j = json.loads(p.read_text()); j["program_tasks"] = sorted([k1, k2]); p.write_text(json.dumps(j))
        g, _ = A.v3_gates(LLM, rows, both, exb)
        assert not g["v3_forced_contributed"], g
        g, _ = A.v3_gates(LLM, rows, full, None)  # no paired replay -> fail closed
        assert not g["v3_forced_contributed"] and not g["v3_forcing_not_negative"]
    print("paired v3 gates ok")


def test_review_gate():
    with tempfile.TemporaryDirectory(dir=A.ROOT / "artifacts/verification/arc-agi-2") as d:
        d = Path(d)
        a, b = d / "a.md", d / "b.md"
        a.write_text("Verdict: GO-WITH-FIXES"); b.write_text("GO")
        rel = lambda p: str(p.relative_to(A.ROOT))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        base = {"verdict": "GO-with-fixes-applied", "kernel_code_change_required": False,
                "reviews": [{"reviewer": "opus", "file": rel(a), "sha256": sha(a), "verdict": "GO-WITH-FIXES"},
                            {"reviewer": "argon", "file": rel(b), "sha256": sha(b), "verdict": "GO"}],
                "p0p1": [{"id": "O1", "status": "FIXED_OUTSIDE_KERNEL", "evidence": "audit_v13.py V2_REPLAY_SHA256"}]}
        r = d / "r.json"

        def ok(obj):
            r.write_text(json.dumps(obj))
            return A.review_go(r, legacy=False)
        assert ok(base)
        assert not ok({**base, "kernel_code_change_required": True})
        assert not ok({**base, "reviews": base["reviews"][:1]})
        assert not ok({**base, "reviews": [base["reviews"][0], {**base["reviews"][1], "verdict": "NO-GO"}]})
        assert not ok({**base, "p0p1": [{"id": "O1", "status": "UNCLEAR", "evidence": "x"}]})
        b.write_text("edited")
        assert not ok(base)  # raw review changed after the verdict file was written
        r.write_text(json.dumps({"verdict": "GO"}))
        assert A.review_go(r, legacy=True) and not A.review_go(r, legacy=False)
    print("review gate ok")


if __name__ == "__main__":
    test_paired_gates()
    test_review_gate()
    print("audit v3 gate tests passed")
