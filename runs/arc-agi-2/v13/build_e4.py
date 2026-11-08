"""Build the E4 Kaggle notebook: does raising the reasoning cap past 64k let the "never-stopped" keys30 tasks finish?
All 12 natural stops of E3 + the v2 dev run came from 7 tasks; the other keys30 tasks never stopped at 64k, and the
cap lever's sign depends on their stop rate q at 80k / 96k (artifacts/verification/arc-agi-2/v13/cap-yield-v1.json,
"stratified"). E4 = one wave of 10 fresh-seed (rep 3) jobs on never-stopped tasks at --cap 96000, no forcing, KV
concurrency formula (-> 10 at 96k). No NVARC; src3 LLM files unchanged (same files as E3). Commit run only, on the
public eval set scored against its solutions; never a submission (the notebook refuses to run as a rerun).
Pre-registration: artifacts/verification/arc-agi-2/v13/prereg-e4-cap96k-v1.md. Never launches or submits.
  .venv/bin/python runs/arc-agi-2/v13/build_e4.py arc-2026-v13-e4-cap96k-v1"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zlib

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402  (shared constants: model / wheelhouse / py312 / image / competition)
import build_e3 as E3  # noqa: E402  (same src3 LLM files and kernel metadata as E3)

ROOT = B.ROOT
SRC = E3.SRC
SRC_FILES = E3.SRC_FILES
JOBS_FILE = "e4-jobs.json"
FILES = SRC_FILES + (JOBS_FILE,)
PROGRAMS = (ROOT / "runs/arc-agi-2/v6/output-arc-2026-v13-e3-seeds-v1/v13-programs.jsonl",
            ROOT / "runs/arc-agi-2/v6/output-arc-2026-v13-progsynth-v2/v13-programs.jsonl")
REP = 3              # fresh seed: crc32("v13:{task}:3"); reps 0-2 were used by v2 / E3
N_JOBS = 10          # one wave at the formula concurrency for cap 96k: int(331,200 / (0.35 * 96000 + 6000)) + 2 = 10
CAP = 96000
E4_SECONDS = 7500    # setup ~650 s + 96k tokens at >= 15 tok/s (E3: 17.8 at conc 13) + abort margin 360 s (Argon P3-5)
SLUG_PREFIX = "arc-2026-v13-e4-"


def never_stopped():
    """keys30 tasks with >= 1 finished (stop/length) job in E3 + v2 and no natural stop in any of them."""
    keys = json.loads((SRC / "keys30.json").read_text())
    done, stopped = set(), set()
    for path in PROGRAMS:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("state") == "done" and r.get("finish_reason") in ("stop", "length"):
                done.add(r["task"])
                if r["finish_reason"] == "stop":
                    stopped.add(r["task"])
    return [k for k in keys if k in done and k not in stopped]


def jobs():
    """N_JOBS never-stopped tasks in a fixed pseudo-random order (crc32 of "e4:<task>"), each once at rep 3."""
    pool = sorted(never_stopped(), key=lambda k: zlib.crc32(f"e4:{k}".encode()))
    return [[k, REP] for k in pool[:N_JOBS]]


def jobs_bytes():
    return (json.dumps(jobs()) + "\n").encode()


def llm_command():
    return f"!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {{e4_end_time}} --jobs {JOBS_FILE} --cap {CAP}"


def notebook(sources):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 E4：提高推理上限（96k）的自然結束率量測（不是提交）\n\n"
        "不跑 NVARC。以 v2 的設定（Qwen3.8-27B-FP8、vLLM 0.19.0、TP4、MTP2、無預算強制），但 cap 96k、併發依 KV 公式，"
        "在公開 eval 的 keys30 中 E3／v2 從未自然結束的題目上跑 10 個新種子（rep 3）工作；每列以 eval 解答評分。"
        "量的是 64k 之後到 80k／96k 之間自然結束的比例（預登記 prereg-e4-cap96k-v1.md）。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in FILES:
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json, os\n"
        "assert not os.getenv('KAGGLE_IS_COMPETITION_RERUN'), 'E4 is an offline measurement kernel, never a submission'\n"
        f"e4_end_time = global_start + {E4_SECONDS}\n"
        "print(json.dumps({'e4_seconds_left': round(e4_end_time - time.time())}))"))
    cells.append(nbf.v4.new_code_cell(llm_command()))
    cells.append(nbf.v4.new_code_cell(
        "with open('/kaggle/working/v13-llm.json') as f:\n    _s = json.load(f)\n"
        "print(json.dumps({k: _s.get(k) for k in ('skipped', 'conc_effective', 'cap', 'gen_seconds', 'summary', 'order')}))"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


def metadata(slug, code_file):
    return E3.metadata(slug, code_file)


def build(slug):
    if not slug.startswith(SLUG_PREFIX):
        raise ValueError("slug must start with " + SLUG_PREFIX)
    out = HERE / f"upload-{slug}"
    if out.exists():
        raise FileExistsError(f"{out} exists; builds are immutable")
    sources = {name: (SRC / name).read_bytes() for name in SRC_FILES}
    sources[JOBS_FILE] = jobs_bytes()
    for name, data in sources.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        else:
            json.loads(data)
    job_list = json.loads(sources[JOBS_FILE])
    if len(job_list) != N_JOBS or len({k for k, _ in job_list}) != N_JOBS or any(r != REP for _, r in job_list):
        raise ValueError("job list: not N_JOBS distinct tasks at rep 3")
    nb = notebook(sources)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata(slug, code_file), indent=2) + "\n")
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": "f0909172434/" + slug, "stage": "v13-e4",
            "purpose": "E4 offline measurement: natural-stop rate past 64k (cap 96k) on never-stopped keys30 tasks "
                       "(never submitted)",
            "approval": "ARC owner 2026-10-15 (user: 「你接下來還有什麼要做要計劃要評估的，都做吧」); 10/17-week ARC share "
                        "40,000 (amendment_2026_10_15d); pre-registration artifacts/verification/arc-agi-2/v13/"
                        "prereg-e4-cap96k-v1.md",
            "algorithm": "v13 v2 program channel (src3, force-at 0) at cap 96k on 10 never-stopped keys30 tasks, rep 3; no NVARC",
            "src_dir": str(SRC.relative_to(ROOT)), "source_sha256": {n: B.sha(d) for n, d in sources.items()},
            "jobs": {"n": len(job_list), "rep": REP, "cap": CAP, "pool": never_stopped(),
                     "programs": [str(p.relative_to(ROOT)) for p in PROGRAMS]},
            "llm_command": llm_command(),
            "notebook_sha256": B.sha((out / code_file).read_bytes()),
            "metadata_sha256": B.sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": E4_SECONDS, "planned_commit_wall_seconds": E4_SECONDS + B.BOOT_OVERHEAD_SECONDS,
            "official_score": None}
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out.relative_to(ROOT)), "kernel": prep["kernel"], "jobs": job_list,
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    build(sys.argv[1])
