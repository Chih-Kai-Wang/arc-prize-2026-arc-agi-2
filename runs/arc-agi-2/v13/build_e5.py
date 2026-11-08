"""Build the E5 Kaggle notebook: one wave of rep-0 keys30 jobs that the v3 dev run did not finish, with v3's own
program-channel settings (pre-registration artifacts/verification/arc-agi-2/v13/prereg-e5-v1.md). Its purpose is the
within-run paired forcing delta (full replay minus `--exclude-source forced`) on more forced jobs; at temperature 1.0 an
E5 job is a fresh draw on that task, not a continuation of the aborted one (vLLM batching diverges long generations).
No NVARC: only the src3 LLM files + `v13_llm.py --jobs e5-jobs.json <v3's llm args>`. Commit run only, on the public
eval set scored against its solutions; never a submission (the notebook refuses to run as a rerun).
Never launches or submits.
  .venv/bin/python runs/arc-agi-2/v13/build_e5.py arc-2026-v13-e5-v3wave-v1 --base arc-2026-v13-progsynth-v3
  .venv/bin/python runs/arc-agi-2/v13/build_e5.py SLUG --base BASE_SLUG [--programs P.jsonl] [--out DIR]   # tests"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402  (shared constants + llm_command flag whitelist)
import build_e3 as E3  # noqa: E402  (same src3 LLM files and kernel metadata as E3)

ROOT = B.ROOT
SRC = E3.SRC
SRC_FILES = E3.SRC_FILES
JOBS_FILE = "e5-jobs.json"
FILES = SRC_FILES + (JOBS_FILE,)
SLUG_PREFIX = "arc-2026-v13-e5-"
E5_SECONDS = 5200       # charge 2.2 x (5,200 + 600) = 12,760 <= the +15,000 of 10/17 (orchestrator 10-16). One wave at
                        # conc 13: setup ~700 s + 58k at >= 17.2 tok/s (v2 wave-1 slowest) = 4,070 s < stop_new_at
                        # (5,200 - 360 - 300 = 4,540) so forcing still starts; + prefill and <= 6k answer tokens
                        # (~400 s) = ~4,470 s < abort_at 4,840 (T3 review 10-16: one wave only, no wave-2 aborts)
MAX_JOBS = 13           # = v3's KV concurrency at 64k (kv 331,200 / (0.35 * 64,000 + 6,000) + 2): exactly one wave
BASES = ("arc-2026-v13-progsynth-v3",)  # branch A only (T3 review 10-16 dropped branch B)


def base_paths(base):
    upload = HERE / f"upload-{base}"
    return upload, ROOT / f"runs/arc-agi-2/v6/output-{base}/v13-programs.jsonl"


def base_llm_args(base):
    if base not in BASES:
        raise ValueError(f"{base}: not an E5 base {BASES}")
    prep = json.loads((base_paths(base)[0] / "preparation.json").read_text())
    if prep.get("src_dir") != str(SRC.relative_to(ROOT)) or prep.get("stage") != "v13":
        raise ValueError(f"{base}: not a src3 build.py dev run")
    return prep["llm_args"]


def finished(r):
    """A base row whose v3 job really ran to its end: the first request stopped, or produced a final-answer program, or
    reached the force point and its forced continuation completed (T3 review 10-16, Argon P0: the first request at
    force_at ends with finish_reason "length"/state "done" even when forcing was then skipped after stop_new_at or the
    continuation was aborted — such a row is unfinished)."""
    if r.get("state") != "done" or r.get("finish_reason") not in ("stop", "length"):
        return False
    if r.get("finish_reason") == "stop" or r.get("n_content_code"):
        return True
    return bool(r.get("forced")) and r.get("cont_state") == "done" and not r.get("force_error")


def jobs(base, programs=None):
    """Rep 0 for the keys30 tasks whose rep-0 job in the base dev run did not finish (aborted, forcing skipped/aborted,
    or never started), in keys30 order with the cut-off ones first; at most one wave (MAX_JOBS)."""
    keys = json.loads((SRC / "keys30.json").read_text())
    rows = [json.loads(line) for line in Path(programs or base_paths(base)[1]).read_text().splitlines() if line.strip()]
    rep0 = {r["task"]: r for r in rows if r.get("rep", 0) == 0}
    unfinished = [k for k in keys if k in rep0 and not finished(rep0[k])]
    unstarted = [k for k in keys if k not in rep0]
    out = [[k, 0] for k in (unfinished + unstarted)[:MAX_JOBS]]
    if not out:
        raise ValueError("nothing to complete")
    return out


def jobs_bytes(base, programs=None):
    return (json.dumps(jobs(base, programs)) + "\n").encode()


def llm_command(llm_args):
    B.llm_command(llm_args)  # same flag whitelist as build.py (v2 none / v3 force / v4 cap [+ force])
    tail = f" {llm_args}" if llm_args else ""
    return f"!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {{e5_end_time}} --jobs {JOBS_FILE}{tail}"


def notebook(sources, llm_args, base):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 E5：v3 設定在 dev run 沒做完的 keys30 題上再跑一波（不是提交）\n\n"
        f"不跑 NVARC。以 `{base}` 的程式通道設定（`{llm_args or '無旗標'}`）在公開 eval 的 keys30 上跑一波 rep 0 工作"
        "（該 dev run 沒做完的題）；每列以 eval 解答評分，在本機以 `replay_v2_merge.py`（完整與排除強制兩版）合併到 E1 HA 三種子，"
        "得同一 run 內的強制配對差。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in FILES:
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json, os\n"
        "assert not os.getenv('KAGGLE_IS_COMPETITION_RERUN'), 'E5 is an offline measurement kernel, never a submission'\n"
        f"e5_end_time = global_start + {E5_SECONDS}\n"
        "print(json.dumps({'e5_seconds_left': round(e5_end_time - time.time())}))"))
    cells.append(nbf.v4.new_code_cell(llm_command(llm_args)))
    cells.append(nbf.v4.new_code_cell(
        "with open('/kaggle/working/v13-llm.json') as f:\n    _s = json.load(f)\n"
        "print(json.dumps({k: _s.get(k) for k in ('skipped', 'conc_effective', 'gen_seconds', 'summary', 'order')}))"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


def build(slug, base, programs=None, out=None):
    if not slug.startswith(SLUG_PREFIX):
        raise ValueError("slug must start with " + SLUG_PREFIX)
    llm_args = base_llm_args(base)
    out = Path(out) if out else HERE / f"upload-{slug}"
    if out.exists():
        raise FileExistsError(f"{out} exists; builds are immutable")
    sources = {name: (SRC / name).read_bytes() for name in SRC_FILES}
    sources[JOBS_FILE] = jobs_bytes(base, programs)
    for name, data in sources.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        else:
            json.loads(data)
    job_list = json.loads(sources[JOBS_FILE])
    nb = notebook(sources, llm_args, base)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    (out / "kernel-metadata.json").write_text(json.dumps(E3.metadata(slug, code_file), indent=2) + "\n")
    base_programs = Path(programs or base_paths(base)[1])
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": "f0909172434/" + slug, "stage": "v13-e5",
            "purpose": f"E5: one wave of the rep-0 keys30 jobs {base} left unfinished (never submitted)",
            "approval": "orchestrator 2026-10-16: +15,000 ARC GPU s in the 10/17 week for offline validation; "
                        "pre-registration artifacts/verification/arc-agi-2/v13/prereg-e5-v1.md",
            "base": base, "llm_args": llm_args, "src_dir": str(SRC.relative_to(ROOT)),
            "source_sha256": {n: B.sha(d) for n, d in sources.items()},
            "jobs": {"n": len(job_list), "tasks": [k for k, _ in job_list],
                     "base_programs": str(base_programs.resolve().relative_to(ROOT)) if base_programs.resolve().is_relative_to(ROOT) else str(base_programs),
                     "base_programs_sha256": B.sha(base_programs.read_bytes())},
            "notebook_sha256": B.sha((out / code_file).read_bytes()),
            "metadata_sha256": B.sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": E5_SECONDS, "planned_commit_wall_seconds": E5_SECONDS + B.BOOT_OVERHEAD_SECONDS,
            "official_score": None}
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out), "kernel": prep["kernel"], "jobs": len(job_list), "llm_args": llm_args,
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--base", required=True)
    ap.add_argument("--programs", default=None)  # tests only: a stand-in programs file
    ap.add_argument("--out", default=None)       # tests only: build outside runs/arc-agi-2/v13
    args = ap.parse_args()
    build(args.slug, args.base, args.programs, args.out)
