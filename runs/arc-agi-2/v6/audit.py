"""Audit a downloaded v6 commit-smoke output. Writes artifacts/verification/arc-agi-2/v6/audit-SLUG.json.

Usage: .venv/bin/python runs/arc-agi-2/v6/audit.py SLUG --hidden-shape-test RESULT.json --review REVIEW.json
submission_allowed is true only when every gate passes. Eval-smoke accuracy is reported, never gated
(the model's pretraining likely saw the public evaluation tasks).
"""
import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E = ROOT / "artifacts/verification/arc-agi-2/v6"
sys.path.insert(0, str(HERE / "src"))
import arc_v6_config as cfg  # noqa: E402


def log_text(out):
    logs = sorted(out.glob("*.log"))
    if not logs:
        return ""
    raw = logs[0].read_text()
    try:
        return "".join(item.get("data", "") for item in json.loads(raw))
    except ValueError:
        return raw


def audit(slug, hidden_shape, review, expect_pass_b=False, expect_v9_plan=False):
    out = HERE / f"output-{slug}"
    gates, facts = {}, {}
    test = json.loads((ROOT / "data/arc-agi-2/official-v1/arc-agi_test_challenges.json").read_text())
    try:
        submission = json.loads((out / "submission.json").read_text())
        gates["submission_full_coverage_valid"] = cfg.validate_submission(test, submission)
    except Exception as exc:
        gates["submission_full_coverage_valid"] = False
        facts["submission_error"] = repr(exc)
    assembly = json.loads((out / "v6-assembly.json").read_text()) if (out / "v6-assembly.json").exists() else {}
    gates["assembly_no_error"] = bool(assembly) and assembly.get("error") is None
    text = log_text(out)
    starts = sorted(set(int(r) for r in re.findall(r"\[Rank (\d)\] start!", text)))
    finished = re.findall(r"\[Rank (\d)\] finished (\S+) in ([\d.]+)s", text)
    failed = re.findall(r"\[Rank (\d)\] task (\S+) failed: (.*)", text)
    alloc = [int(x) for x in re.findall(r"allocated (\d+)MB", text)]
    exits = re.findall(r'"workers_exited", "exitcodes": (\[[^\]]*\])', text)
    smoke = re.findall(r'"eval_smoke_score", "solved": ([\d.]+), "tasks": (\d+)', text)
    facts.update(ranks_started=starts, tasks_finished=[{"rank": int(r), "task": k, "seconds": float(s)} for r, k, s in finished],
                 task_failures=[{"rank": int(r), "task": k, "error": e[:300]} for r, k, e in failed],
                 peak_allocated_mb=max(alloc) if alloc else None, worker_exitcodes=json.loads(exits[-1]) if exits else None,
                 eval_smoke=({"solved": float(smoke[-1][0]), "tasks": int(smoke[-1][1]),
                              "contaminated_by_pretraining": True} if smoke else None),
                 decoded_files=assembly.get("decoded_files"))
    gates["four_workers_started"] = starts == [0, 1, 2, 3]
    gates["smoke_tasks_finished"] = len({k for _, k, _ in finished}) >= cfg.SMOKE_TASKS
    gates["no_task_failures"] = not failed
    # v8: one workers_exited line per pass; every pass must exit cleanly.
    gates["workers_exited_cleanly"] = bool(exits) and all(json.loads(e) == [0, 0, 0, 0] for e in exits)
    if expect_pass_b:
        b_at = text.find('"pass_b_start"')
        # Pass B's own window ends at its workers_exited line (assembly/benchmark output follows it).
        b_end = text.find('"workers_exited"', b_at) if b_at >= 0 else -1
        b_text = text[b_at:b_end] if b_at >= 0 and b_end >= 0 else (text[b_at:] if b_at >= 0 else "")
        finished_b = re.findall(r"\[Rank (\d)\] finished (\S+) in ([\d.]+)s", b_text)
        solved_a = re.findall(r'"eval_smoke_score".*?"solved_pass_a": ([\d.]+)', text)
        facts["pass_b"] = {"started": b_at >= 0, "skipped": '"pass_b_skipped"' in text,
                           "tasks_finished": [{"rank": int(r), "task": k, "seconds": float(s)} for r, k, s in finished_b],
                           "worker_exitcodes_per_pass": [json.loads(e) for e in exits],
                           "eval_smoke_solved_pass_a": float(solved_a[-1]) if solved_a else None,
                           **{k: assembly.get(k) for k in ("outputs_a", "outputs_pooled", "pass_b_files", "changed_vs_pass_a")}}
        # Seeds took effect: pass-B decode subkeys carry permutations disjoint from pass A's.
        perms_a = set(re.findall(r"permute(\d{10})", text[:b_at])) if b_at >= 0 else set()
        perms_b = set(re.findall(r"permute(\d{10})", b_text))
        b_starts = sorted(set(int(r) for r in re.findall(r"\[Rank (\d)\] start! pass B", text)))
        facts["pass_b"].update(ranks_started=b_starts, permutations_a=len(perms_a), permutations_b=len(perms_b),
                               permutations_shared=len(perms_a & perms_b))
        gates["pass_b_ran"] = (b_at >= 0 and len(exits) == 2 and b_starts == [0, 1, 2, 3] and bool(finished_b)
                               and (assembly.get("pass_b_files") or 0) > 0)
        gates["pass_b_seeds_differ"] = bool(perms_a) and bool(perms_b) and not perms_a & perms_b
        try:  # v9: the pass_b_start line carries the pass-B plan (confidence order or v8 fallback)
            plan = json.loads(text[text.rfind("{", 0, b_at):text.find("\n", b_at)].strip()) if b_at >= 0 else None
        except ValueError:
            plan = None
        facts["pass_b"]["plan"] = plan
        if expect_v9_plan:
            gates["pass_b_plan_confidence"] = bool(plan) and plan.get("pass_b_order") == "confidence"
        gates["pooling_no_error"] = bool(assembly) and "pooling_error" not in assembly
    gates["model_produced_outputs"] = (assembly.get("decoded_files") or 0) > 0
    hs = json.loads(Path(hidden_shape).read_text())
    gates["hidden_shape_test_passed"] = hs.get("passed") is True
    rv = json.loads(Path(review).read_text())
    gates["independent_review_go"] = rv.get("verdict") in ("GO", "GO-with-fixes-applied")
    result = {"kernel": "f0909172434/" + slug, "gates": gates, "facts": facts,
              "hidden_shape_test": str(Path(hidden_shape).resolve().relative_to(ROOT)),
              "review": str(Path(review).resolve().relative_to(ROOT)),
              "submission_allowed": all(gates.values()), "official_score": None}
    path = E / f"audit-{slug}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"submission_allowed": result["submission_allowed"],
                      "failed_gates": [k for k, v in gates.items() if not v], "eval_smoke": facts["eval_smoke"],
                      "seconds": [t["seconds"] for t in facts["tasks_finished"]], "peak_mb": facts["peak_allocated_mb"]}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("slug")
    p.add_argument("--hidden-shape-test", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--expect-pass-b", action="store_true", help="v8+: gate on the pass-B smoke facts")
    p.add_argument("--expect-v9-plan", action="store_true", help="v9+: gate on pass B using the confidence plan")
    a = p.parse_args()
    need_b = a.expect_pass_b or a.expect_v9_plan
    audit(a.slug, a.hidden_shape_test, a.review, need_b, a.expect_v9_plan)
