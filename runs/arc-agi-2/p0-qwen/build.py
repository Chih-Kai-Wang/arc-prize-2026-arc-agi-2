"""Build the P0 probe kernel folder (new-approach-proposal-v1 §6): one notebook whose %%writefile cells carry src/,
then runs the probe. Records sha256 of every source in preparation.json for the launch gate.
  .venv/bin/python runs/arc-agi-2/p0-qwen/build.py"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
SLUG = "arc-2026-p0-qwen38-v2"  # v1 (2026-10-12) failed fast: pinned image had Python 3.11, wheelhouse is cp312
OUT = HERE / f"upload-{SLUG}"
FILES = ["arc_llm.py", "p0_probe.py", "keys30.json", "qwen38_fp8_sha256.json"]
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"  # Python 3.12.13 (sota-v1..v5 startup.json)
MODEL = "mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1"
WHEELS = "mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
WALL = int(3.5 * 3600)  # commit wall cap used for the GPU reservation; the probe stops itself at 3 h 10 min


def sha(b):
    return hashlib.sha256(b).hexdigest()


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    OUT.mkdir(exist_ok=True)
    cells = [cell("markdown", "# ARC 2026 P0 probe: Qwen3.8-27B-FP8 on 4xL4\nDev run only (not a submission). See "
                              "artifacts/verification/arc-agi-2/new-approach-proposal-v1.md §6.\n")]
    srcs = {}
    for name in FILES:
        body = (SRC / name).read_text()
        srcs[name] = sha(body.encode())
        cells.append(cell("code", f"%%writefile {name}\n{body}"))
    cells.append(cell("code", "!cd /kaggle/working && python p0_probe.py\n"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    nb_bytes = json.dumps(nb, indent=1).encode()
    (OUT / f"{SLUG}.ipynb").write_bytes(nb_bytes)
    meta = {"id": f"f0909172434/{SLUG}", "title": SLUG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "dataset_sources": [WHEELS], "kernel_sources": [], "model_sources": [MODEL], "docker_image": IMAGE,
            "machine_shape": "NvidiaL4"}
    meta_bytes = json.dumps(meta, indent=1).encode()
    (OUT / "kernel-metadata.json").write_bytes(meta_bytes)
    prep = {"kernel": meta["id"], "notebook_sha256": sha(nb_bytes), "metadata_sha256": sha(meta_bytes),
            "source_sha256": srcs, "src_dir": "runs/arc-agi-2/p0-qwen/src", "planned_commit_wall_seconds": WALL}
    (OUT / "preparation.json").write_text(json.dumps(prep, indent=1) + "\n")
    print(json.dumps(prep, indent=1))


if __name__ == "__main__":
    main()
