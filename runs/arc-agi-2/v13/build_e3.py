"""Build the E3 Kaggle notebook: two more independent seeds of the v13 v2 program channel on keys30, for the slot-1
offline mean (final_selection_amendment_2026_10_13). No NVARC: only the src3 LLM files + `v13_llm.py --jobs e3-jobs.json`
(force-at 0 = v2 behaviour; cap 64k, KV concurrency formula, prompts, verification all as v2). Commit run only, on the
public eval set scored against its solutions; it is never a submission (the notebook refuses to run as a rerun).
Pre-registration: artifacts/verification/arc-agi-2/v13/prereg-e3-seeds-v1.md. Never launches or submits.
  .venv/bin/python runs/arc-agi-2/v13/build_e3.py arc-2026-v13-e3-seeds-v1"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402  (shared constants: model / wheelhouse / py312 / image / competition)

ROOT = B.ROOT
SRC = HERE / "src3"
SRC_FILES = ("arc_llm.py", "progsynth.py", "v13_merge.py", "v13_llm.py", "keys30.json", "qwen38_fp8_sha256.json")
JOBS_FILE = "e3-jobs.json"
FILES = SRC_FILES + (JOBS_FILE,)
V2_PROGRAMS = ROOT / "runs/arc-agi-2/v6/output-arc-2026-v13-progsynth-v2/v13-programs.jsonl"
SEED_REPS = (1, 2)
E3_SECONDS = 20000  # setup ~650 s + 60-job makespan p95 18,218 s (v2 wave-1 durations, conc 13) + abort margin 360 s
SLUG_PREFIX = "arc-2026-v13-e3-"


def jobs():
    """Interleaved (task, 1), (task, 2) over keys30 in list order (if time runs short both seeds lose the same tasks),
    then a tail of rep 0 for the keys30 tasks whose v2 rep-0 job did not finish (aborted / never started): it only
    uses slots left over in the last wave and is reported separately, never averaged into the seed mean."""
    keys = json.loads((SRC / "keys30.json").read_text())
    done = set()
    for line in V2_PROGRAMS.read_text().splitlines():
        r = json.loads(line)
        if r.get("rep") == 0 and r.get("state") == "done" and r.get("finish_reason") in ("stop", "length"):
            done.add(r["task"])
    return [[k, r] for k in keys for r in SEED_REPS] + [[k, 0] for k in keys if k not in done]


def jobs_bytes():
    return (json.dumps(jobs()) + "\n").encode()


def notebook(sources):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 E3：v13 程式通道多種子離線量測（不是提交）\n\n"
        "不跑 NVARC。以 v2 的設定（Qwen3.8-27B-FP8、vLLM 0.19.0、TP4、MTP2、cap 64k、無預算強制）在公開 eval 的 keys30 上"
        "跑程式通道 rep 1 與 rep 2（兩個獨立種子），尾段補 v2 未完成的 rep 0；每列以 eval 解答評分。"
        "結果在本機以 `replay_v2_merge.py --rep N` 合併到 E1 HA 三種子，得第 1 名額要用的離線均值。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in FILES:
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json, os\n"
        "assert not os.getenv('KAGGLE_IS_COMPETITION_RERUN'), 'E3 is an offline measurement kernel, never a submission'\n"
        f"e3_end_time = global_start + {E3_SECONDS}\n"
        "print(json.dumps({'e3_seconds_left': round(e3_end_time - time.time())}))"))
    cells.append(nbf.v4.new_code_cell(f"!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {{e3_end_time}} --jobs {JOBS_FILE}"))
    cells.append(nbf.v4.new_code_cell(
        "with open('/kaggle/working/v13-llm.json') as f:\n    _s = json.load(f)\n"
        "print(json.dumps({k: _s.get(k) for k in ('skipped', 'conc_effective', 'gen_seconds', 'summary', 'order')}))"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


def metadata(slug, code_file):
    return {"id": "f0909172434/" + slug, "title": slug.replace("-", " "), "code_file": code_file,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "competition_sources": [B.COMPETITION],
            "dataset_sources": [B.WHEELS], "kernel_sources": [B.PY312_KERNEL], "model_sources": [B.QWEN38],
            "docker_image": B.IMAGE, "machine_shape": "NvidiaL4"}


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
    if len(job_list) != len({tuple(j) for j in job_list}) or sum(r in SEED_REPS for _, r in job_list) != 60:
        raise ValueError("job list: duplicates or not 30 tasks x 2 seeds")
    nb = notebook(sources)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata(slug, code_file), indent=2) + "\n")
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": "f0909172434/" + slug, "stage": "v13-e3",
            "purpose": "E3 offline seeds of the v13 v2 program channel on keys30 (never submitted)",
            "approval": "orchestrator 2026-10-14: E3 on Kaggle 4xL4 (GCP infeasible), in the 10/17 week; "
                        "pre-registration artifacts/verification/arc-agi-2/v13/prereg-e3-seeds-v1.md",
            "algorithm": "v13 v2 program channel (src3, force-at 0) on keys30, reps 1+2 interleaved, rep-0 tail; no NVARC",
            "src_dir": str(SRC.relative_to(ROOT)), "source_sha256": {n: B.sha(d) for n, d in sources.items()},
            "jobs": {"n": len(job_list), "seed_reps": list(SEED_REPS),
                     "rep0_tail": [k for k, r in job_list if r == 0], "v2_programs": str(V2_PROGRAMS.relative_to(ROOT))},
            "notebook_sha256": B.sha((out / code_file).read_bytes()),
            "metadata_sha256": B.sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": E3_SECONDS, "planned_commit_wall_seconds": E3_SECONDS + B.BOOT_OVERHEAD_SECONDS,
            "official_score": None}
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out.relative_to(ROOT)), "kernel": prep["kernel"], "jobs": len(job_list),
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    build(sys.argv[1])
