"""Build the v6 Kaggle notebook (author 4B pipeline + v6 hardening). Never launches or submits.

Usage: .venv/bin/python runs/arc-agi-2/v6/build.py SLUG [--src DIR]
Writes <DIR>/../upload-SLUG/{SLUG.ipynb, kernel-metadata.json, preparation.json} (default DIR = v6/src).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = HERE / "src"
FILES = ("arc_v6_config.py", "arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py", "assemble.py")
MODEL = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
UNSLOTH = "sorokin/pip-install-unsloth-flash-patch"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
COMPETITION = "arc-prize-2026-arc-agi-2"
BOOT_OVERHEAD_SECONDS = 600  # docker pull/boot + model load before the in-notebook clock bounds work


def sha(data):
    return hashlib.sha256(data).hexdigest()


def notebook(sources):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 v6：作者 Qwen3-4B 格子模型流程（加固版）\n\n"
        "演算法與 [sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue) "
        "相同：每題 rank-256 rsLoRA 測試時訓練（128 條增強序列）、16 視角 turbo-DFS、增強重評分。\n\n"
        "v6 只加穩健性：路徑自動偵測、commit 跑 4 題冒煙／評分重跑跑全部題目、開始前先寫完整備援 submission.json、"
        "單題例外隔離、worker 互不連坐、原子寫檔、到期強制收尾。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!pip uninstall -y tensorflow"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in FILES:
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json\nimport arc_v6_config as cfg\n"
        "global_end_time = cfg.end_time(global_start)\n"
        "with open(cfg.test_path()) as f:\n    _test = json.load(f)\n"
        "cfg.atomic_json(cfg.WORK_DIR + '/submission.json', cfg.fallback_submission(_test))\n"
        "cfg.log('fallback_written', rerun=cfg.is_rerun(), test_tasks=len(_test), model_dir=cfg.model_dir(),\n"
        "        competition_dir=cfg.competition_dir(),\n"
        "        work_file=cfg.work_challenges_path(), seconds_left=round(global_end_time - time.time()))"))
    cells.append(nbf.v4.new_code_cell(
        "!PYTHONUNBUFFERED=1 UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 "
        "python starter.py --end-time {global_end_time}"))
    cells.append(nbf.v4.new_code_cell("import assemble\nrecord = assemble.main()"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


def build(slug, src=SRC):
    src = Path(src).resolve()
    if not slug.startswith("arc-2026-"):
        raise ValueError("slug must start with arc-2026-")
    out = src.parent / f"upload-{slug}"
    if out.exists():
        raise FileExistsError(f"{out} exists; builds are immutable")
    sources = {name: (src / name).read_bytes() for name in FILES}
    for name, data in sources.items():
        compile(data, name, "exec")
    nb = notebook(sources)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    meta = {"id": "f0909172434/" + slug, "title": slug.replace("-", " "), "code_file": code_file,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "competition_sources": [COMPETITION],
            "dataset_sources": [], "kernel_sources": [UNSLOTH], "model_sources": [MODEL],
            "docker_image": IMAGE, "machine_shape": "NvidiaL4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    sys.path.insert(0, str(src))
    import arc_v6_config as cfg
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": meta["id"], "stage": "v6",
            "algorithm": "author sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue v10 (unchanged search/TTT/scoring)",
            "src_dir": str(src.relative_to(ROOT)),
            "source_sha256": {name: sha(data) for name, data in sources.items()},
            "notebook_sha256": sha((out / code_file).read_bytes()),
            "metadata_sha256": sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": cfg.COMMIT_WALL_SECONDS, "rerun_wall_seconds": cfg.RERUN_WALL_SECONDS,
            "planned_commit_wall_seconds": cfg.COMMIT_WALL_SECONDS + BOOT_OVERHEAD_SECONDS,
            "push_timeout": None, "official_score": None}
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out.relative_to(ROOT)), "kernel": meta["id"],
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--src", type=Path, default=SRC)
    args = parser.parse_args()
    build(args.slug, args.src)
