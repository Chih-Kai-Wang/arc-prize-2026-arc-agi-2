"""Audit a downloaded commit output of the AS-IS perfpatch fork. Writes artifacts/verification/arc-agi-2/v6/audit-SLUG.json.

Usage: python runs/arc-agi-2/fork-perfpatch/audit_fork.py arc-2026-fork-perfpatch-v1 \
           --hidden-shape-test RESULT.json --review REVIEW.json
Reads runs/arc-agi-2/v6/output-SLUG/ (cloud.py download). Commit mode of the notebook runs only 4 evaluation tasks through
the workers, but its final cell builds submission.json from ALL evaluation challenges (unsolved ones keep [[0]]), so the
expected id set is every evaluation task id. Eval accuracy is reported, never gated (pretraining likely saw eval tasks).
submission_allowed is true only when every gate passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT_ROOT = ROOT / "runs/arc-agi-2/v6"
E = ROOT / "artifacts/verification/arc-agi-2/v6"
EVAL = ROOT / "data/arc-agi-2/official-v1/arc-agi_evaluation_challenges.json"
UPLOAD = HERE / "upload-arc-2026-fork-perfpatch-v1"
PINNED_NOTEBOOK_SHA256 = "ee49384a10a4296dc0bf0670ae67325521f798d5337ed57bee50c8a443a378f4"  # mikelou1 perfpatch, hard-coded
EXTERNAL_SRC = OUTPUT_ROOT / "external-src/mikelou1__arc-agi2-lb33-89-minimal-perfpatch/arc-agi2-lb33-89-minimal-perfpatch.ipynb"
SRC_DIR = HERE / "src"
BUILD = OUTPUT_ROOT / "build.py"
COMPETITION = "arc-prize-2026-arc-agi-2"
SMOKE_IDS = ("0934a4d8", "36a08778", "981571dc", "aa4ec2a5")  # the notebook's own commit-mode task list


def log_text(out):
    """Concatenate every *.log in the output dir (Kaggle may name the notebook log differently)."""
    parts = []
    for log in sorted(Path(out).glob("*.log")):
        raw = log.read_text()
        try:
            parts.append("".join(item.get("data", "") for item in json.loads(raw)))
        except (ValueError, AttributeError):
            parts.append(raw)
    return "\n".join(parts)


def build_constants(path=BUILD):
    """MODEL / UNSLOTH / IMAGE string constants from v6/build.py, read without importing it."""
    import ast
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("MODEL", "UNSLOTH", "IMAGE"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_grid(g):
    return (isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) > 0 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(x) is int and 0 <= x <= 9 for r in g for x in r))


def validate_submission(challenges, submission):
    if not isinstance(submission, dict) or set(submission) != set(challenges):
        return False
    for tid, task in challenges.items():
        entries = submission[tid]
        if not isinstance(entries, list) or len(entries) != len(task["test"]):
            return False
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                return False
            if not (valid_grid(entry["attempt_1"]) and valid_grid(entry["attempt_2"])):
                return False
    return True


def audit(slug, hidden_shape, review, out=None, eval_path=EVAL, upload=UPLOAD, e_dir=E, pinned_sha=PINNED_NOTEBOOK_SHA256,
          external=EXTERNAL_SRC, src_dir=SRC_DIR, build=BUILD):
    out = Path(out) if out else OUTPUT_ROOT / f"output-{slug}"
    gates, facts = {}, {}
    prep = json.loads((Path(upload) / "preparation.json").read_text())
    notebook = Path(upload) / json.loads((Path(upload) / "kernel-metadata.json").read_text())["code_file"]
    upload_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
    facts.update(upload_notebook_sha256=upload_sha, source_notebook=prep["source_notebook"],
                 source_notebook_sha256=prep["source_notebook_sha256"])
    external_sha = sha256_file(external) if Path(external).exists() else None
    facts["external_src_notebook_sha256"] = external_sha
    # pinned constant + both files hashed now; preparation.json is recorded as a fact only
    gates["byte_identical_fork"] = upload_sha == pinned_sha and external_sha == pinned_sha
    meta = json.loads((Path(upload) / "kernel-metadata.json").read_text())
    const = build_constants(build)
    gates["metadata_private_offline_l4"] = (meta.get("enable_internet") is False and meta.get("is_private") is True
                                            and meta.get("machine_shape") == "NvidiaL4")
    gates["metadata_sources_match_build"] = (meta.get("competition_sources") == [COMPETITION]
                                             and meta.get("model_sources") == [const.get("MODEL")]
                                             and meta.get("kernel_sources") == [const.get("UNSLOTH")]
                                             and meta.get("docker_image") == const.get("IMAGE") and len(const) == 3)
    challenges = json.loads(Path(eval_path).read_text())
    try:
        submission = json.loads((out / "submission.json").read_text())
        gates["submission_full_coverage_valid"] = validate_submission(challenges, submission)
        gates["smoke_ids_have_real_attempt"] = all(
            isinstance(submission.get(t), list) and submission[t]
            and all(isinstance(e, dict) and any(e.get(k) != [[0]] for k in ("attempt_1", "attempt_2")) for e in submission[t])
            for t in SMOKE_IDS)
    except Exception as exc:
        gates["submission_full_coverage_valid"] = False
        gates["smoke_ids_have_real_attempt"] = False
        facts["submission_error"] = repr(exc)
    facts["expected_task_ids"] = len(challenges)
    text = log_text(out)
    starts = sorted(set(int(r) for r in re.findall(r"\[Rank (\d)\] start!", text)))
    finished = re.findall(r"\[Rank (\d)\] finished (\S+) in ([\d.]+)s", text)
    timeouts = re.findall(r"\[Rank (\d)\] timeout after ([\d.]+)s for puzzle (\S+)", text)
    tracebacks = re.findall(r"Traceback \(most recent call last\)", text)
    alloc = [int(x) for x in re.findall(r"allocated (\d+)MB", text)]
    reload_score = re.findall(r"\*\*\* Reload score: ([\d.]+)", text)
    facts.update(ranks_started=starts, tasks_finished=[{"rank": int(r), "task": k, "seconds": float(s)} for r, k, s in finished],
                 task_timeouts=[{"rank": int(r), "task": k, "seconds": float(s)} for r, s, k in timeouts],
                 traceback_count=len(tracebacks), peak_allocated_mb=max(alloc) if alloc else None,
                 eval_reload_score=float(reload_score[-1]) if reload_score else None,
                 scoring_lines=len(re.findall(r"\[Rank \d\] scoring ", text)))
    gates["four_workers_started"] = starts == [0, 1, 2, 3]
    # The notebook has no per-task failure line (no try/except); a failing worker surfaces as a Traceback.
    gates["no_task_failures"] = not tracebacks
    gates["smoke_tasks_finished"] = {k for _, k, _ in finished} >= set(SMOKE_IDS)
    gates["final_assembly_completed"] = bool(reload_score)  # printed only after submission.json was written and reloaded
    gates["reload_score_positive"] = bool(reload_score) and float(reload_score[-1]) > 0
    # decoded-result files (/kaggle/inference_outputs) are not normally in the download; gate only if present
    decoded = [p for p in out.rglob("*") if p.is_file() and p.parent.name == "inference_outputs"]
    decoded_ids = {p.name.split(".")[0].split("_")[0] for p in decoded}
    facts["decoded_files_in_download"] = len(decoded)
    facts["decoded_smoke_ids"] = sorted(decoded_ids & set(SMOKE_IDS))
    if decoded:
        gates["decoded_files_cover_smoke_ids"] = set(SMOKE_IDS) <= decoded_ids
    facts["model_licence"] = "Apache 2.0 (Kaggle model sorokin/qwen3_4b_grids15_sft139 bfloat16 instance licenseName, read 2026-10-12T21:30Z)"
    hs = json.loads(Path(hidden_shape).read_text())
    current_src = {p.name: sha256_file(p) for p in sorted(Path(src_dir).glob("*.py"))}
    gates["hidden_shape_test_passed"] = (hs.get("passed") is True and hs.get("src_sha256") == current_src
                                         and hs.get("notebook_sha256") == upload_sha)
    rv = json.loads(Path(review).read_text())
    gates["independent_review_go"] = rv.get("verdict") in ("GO", "GO-with-fixes-applied")

    def rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(Path(p).resolve())

    result = {"kernel": "f0909172434/" + slug, "gates": gates, "facts": facts,
              "hidden_shape_test": rel(hidden_shape), "review": rel(review),
              "submission_allowed": all(gates.values()), "official_score": None}
    path = Path(e_dir) / f"audit-{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"submission_allowed": result["submission_allowed"],
                      "failed_gates": [k for k, v in gates.items() if not v], "eval_reload_score": facts["eval_reload_score"],
                      "seconds": [t["seconds"] for t in facts["tasks_finished"]], "peak_mb": facts["peak_allocated_mb"]}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("slug")
    p.add_argument("--hidden-shape-test", required=True)
    p.add_argument("--review", required=True)
    a = p.parse_args()
    audit(a.slug, a.hidden_shape_test, a.review)
