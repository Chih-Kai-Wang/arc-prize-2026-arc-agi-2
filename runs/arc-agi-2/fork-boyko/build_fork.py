"""Build the Boyko V22 fork upload: two commit-mode-or-default edits only, recorded in preparation.json.
  .venv/bin/python runs/arc-agi-2/fork-boyko/build_fork.py"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXT = ROOT / "runs/arc-agi-2/v6/external-src/evgendvorkin__arc-2-2026-4-4-local-boyko-nvarc"
SRC_NB = EXT / "arc-2-2026-4-4-local-boyko-nvarc.ipynb"
SLUG = "arc-2026-fork-boyko-v1"
UP = HERE / f"upload-{SLUG}"
SOURCE_SHA = "3869a26872c6bb57a9b69809bdaa3ee62c0593eafd905943493add6b7d4eb46c"  # evgendvorkin V22 as pulled 10-31
EDITS = [  # (cell index, old, new, why)
    (5, "MAX_SECONDS = 12 * 3600 - 900 if RERUN_MODE else 75 * 60",
        "MAX_SECONDS = 12 * 3600 - 900 if RERUN_MODE else 25 * 60",
        "commit-mode budget only (GPU quota); rerun branch unchanged"),
    (16, "promote=os.getenv('ARC_ASTRA_PROMOTE','1')=='1'",
         "promote=os.getenv('ARC_ASTRA_PROMOTE','0')=='1'",
         "expert promotion off by default = author's scored V17-V20 pure-neural A+B selection"),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    raw = SRC_NB.read_bytes()
    assert sha(raw) == SOURCE_SHA, "source notebook changed"
    nb = json.loads(raw)
    for i, old, new, _ in EDITS:
        src = "".join(nb["cells"][i]["source"])
        assert src.count(old) == 1, (i, old)
        src = src.replace(old, new)
        nb["cells"][i]["source"] = src.splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    UP.mkdir(exist_ok=True)
    out = UP / f"{SLUG}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    src_meta = json.loads((EXT / "kernel-metadata.json").read_text())
    meta = {"id": f"f0909172434/{SLUG}", "title": SLUG.replace("-", " "), "code_file": out.name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": src_meta["dataset_sources"],
            "kernel_sources": src_meta["kernel_sources"], "competition_sources": src_meta["competition_sources"],
            "model_sources": src_meta["model_sources"], "docker_image": src_meta["docker_image"],
            "machine_shape": src_meta["machine_shape"]}
    (UP / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    srcdir = HERE / "src"
    srcdir.mkdir(exist_ok=True)
    files = {}
    for c in nb["cells"]:
        s = "".join(c["source"])
        m = re.match(r"%%writefile (\S+)\n", s)
        if c["cell_type"] == "code" and m:
            (srcdir / m.group(1)).write_text(s[m.end():])
            files[m.group(1)] = sha((srcdir / m.group(1)).read_bytes())
    prep = {"slug": SLUG, "kernel": meta["id"],
            "purpose": "Boyko V22 fork (promote off = V17-V20 A+B) commit run + slot-2 candidate",
            "approval": "ARC owner per user 11-01 「讓arc全力推進」 (coordinator relay); prereg "
                        "artifacts/verification/arc-agi-2/fork-boyko/prereg-boyko-fork-v1.md", "source_notebook": "evgendvorkin/arc-2-2026-4-4-local-boyko-nvarc (V22)",
            "source_sha256": SOURCE_SHA, "notebook_sha256": sha(out.read_bytes()),
            "metadata_sha256": sha((UP / "kernel-metadata.json").read_bytes()),
            "edits": [{"cell": i, "old": o, "new": n, "why": w} for i, o, n, w in EDITS], "src_sha256": files,
            "planned_commit_wall_seconds": 2700}
    (UP / "preparation.json").write_text(json.dumps(prep, indent=1) + "\n")
    print(json.dumps({k: prep[k] for k in ("notebook_sha256", "metadata_sha256")}))


if __name__ == "__main__":
    main()
