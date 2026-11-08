"""Build the E6 Kaggle notebook: two independent seeds of the v4-c96 program channel (cap 96k, no forcing) on keys30,
so v4 gets the multi-repeat offline mean that slot 1 requires (prereg-v4-v1.md §2; final_selection_amendment_2026_10_13).
Same design as E3 (src3 LLM files, keys30 reps 1+2 interleaved, scored against the eval solutions, no NVARC) with the
v4-c96 call (`--cap 96000`, KV concurrency formula -> 10) and no rep-0 tail. Commit run only; never a submission (the
notebook refuses to run as a rerun). Pre-registration: artifacts/verification/arc-agi-2/v13/prereg-e6-v4-seeds-v1.md.
Never launches or submits.
  .venv/bin/python runs/arc-agi-2/v13/build_e6.py arc-2026-v13-e6-v4seeds-v1"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402  (shared constants: model / wheelhouse / py312 / image / competition)
import build_e3 as E3  # noqa: E402  (same src3 LLM files and kernel metadata as E3)

ROOT = B.ROOT
SRC = E3.SRC
SRC_FILES = E3.SRC_FILES
JOBS_FILE = "e6-jobs.json"
FILES = SRC_FILES + (JOBS_FILE,)
SEED_REPS = (1, 2)   # same reps (seeds and augmentation salts) as E3, so E6 rep r pairs with E3 rep r
CAP = 96000
# setup ~650 s + 60 jobs / conc 10 = 6 waves x <= 5,500 s (v4-c96 dev run: 96k jobs aborted at 89-92k after ~5,300 s;
# E4 median > 64k decode 18.55 tok/s) + abort margin 360 s
E6_SECONDS = 35000
SLUG_PREFIX = "arc-2026-v13-e6-"
metadata = E3.metadata


def jobs():
    """Interleaved (task, 1), (task, 2) over keys30 in list order, as E3 (if time runs short both seeds lose the same
    tasks). No rep-0 tail."""
    keys = json.loads((SRC / "keys30.json").read_text())
    return [[k, r] for k in keys for r in SEED_REPS]


def jobs_bytes():
    return (json.dumps(jobs()) + "\n").encode()


def llm_command():
    return f"!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {{e6_end_time}} --jobs {JOBS_FILE} --cap {CAP}"


def notebook(sources):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 E6：v4-c96 程式通道兩種子離線量測（不是提交）\n\n"
        "不跑 NVARC。以 v4-c96 的設定（Qwen3.8-27B-FP8、vLLM 0.19.0、TP4、MTP2、cap 96k、無預算強制、併發照 KV 公式）"
        "在公開 eval 的 keys30 上跑程式通道 rep 1 與 rep 2（與 E3 同種子）；每列以 eval 解答評分。"
        "結果在本機以 `replay_v2_merge.py --rep N` 合併到 E1 HA 三種子，得 v4 的多次離線均值。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in FILES:
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json, os\n"
        "assert not os.getenv('KAGGLE_IS_COMPETITION_RERUN'), 'E6 is an offline measurement kernel, never a submission'\n"
        f"e6_end_time = global_start + {E6_SECONDS}\n"
        "print(json.dumps({'e6_seconds_left': round(e6_end_time - time.time())}))"))
    cells.append(nbf.v4.new_code_cell(llm_command()))
    cells.append(nbf.v4.new_code_cell(
        "with open('/kaggle/working/v13-llm.json') as f:\n    _s = json.load(f)\n"
        "print(json.dumps({k: _s.get(k) for k in ('skipped', 'cap', 'conc_effective', 'gen_seconds', 'summary', 'order')}))"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


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
    if len(job_list) != 60 or len({tuple(j) for j in job_list}) != 60 or {r for _, r in job_list} != set(SEED_REPS):
        raise ValueError("job list: not 30 tasks x 2 seeds without duplicates")
    nb = notebook(sources)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata(slug, code_file), indent=2) + "\n")
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": "f0909172434/" + slug, "stage": "v13-e6",
            "purpose": "E6 offline seeds of the v4-c96 program channel (cap 96k) on keys30 (never submitted)",
            "approval": "ARC owner session (user 10-17 「Go，以後都自己決定」); reserve per orchestrator 10-17; "
                        "pre-registration artifacts/verification/arc-agi-2/v13/prereg-e6-v4-seeds-v1.md",
            "algorithm": "v4-c96 program channel (src3, force-at 0, --cap 96000) on keys30, reps 1+2 interleaved; no NVARC",
            "src_dir": str(SRC.relative_to(ROOT)), "source_sha256": {n: B.sha(d) for n, d in sources.items()},
            "jobs": {"n": len(job_list), "seed_reps": list(SEED_REPS), "cap": CAP},
            "notebook_sha256": B.sha((out / code_file).read_bytes()),
            "metadata_sha256": B.sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": E6_SECONDS, "planned_commit_wall_seconds": E6_SECONDS + B.BOOT_OVERHEAD_SECONDS,
            "official_score": None}
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out.relative_to(ROOT)), "kernel": prep["kernel"], "jobs": len(job_list),
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    build(sys.argv[1])
