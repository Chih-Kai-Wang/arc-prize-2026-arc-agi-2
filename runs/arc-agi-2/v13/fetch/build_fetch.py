"""Build the internet-enabled CPU kernel that downloads python-build-standalone cpython-3.12.14+20260924 (x86_64 linux,
install_only) from astral's release mirror, checks its sha256 and leaves it in /kaggle/working, so v13 can attach the
output as a kernel source (v13 itself stays offline). Replaces the private dataset, whose creation failed silently on
2026-10-13. Push: kaggle kernels push -p runs/arc-agi-2/v13/fetch/upload-arc-2026-py312-fetch-v1
  .venv/bin/python runs/arc-agi-2/v13/fetch/build_fetch.py"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "arc-2026-py312-fetch-v1"
OUT = HERE / f"upload-{SLUG}"
NAME = "cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"
SHA = "5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682"
URLS = ["https://releases.astral.sh/github/python-build-standalone/releases/download/20260924/"
        "cpython-3.12.14%2B20260924-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
        "cpython-3.12.14%2B20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"]
CODE = f'''import hashlib, json, urllib.request
NAME, SHA, URLS = {NAME!r}, {SHA!r}, {URLS!r}
rec = {{"name": NAME, "sha256_expected": SHA, "tries": []}}
for url in URLS:
    try:
        data = urllib.request.urlopen(url, timeout=600).read()
        h = hashlib.sha256(data).hexdigest()
        rec["tries"].append({{"url": url, "bytes": len(data), "sha256": h}})
        if h == SHA:
            open("/kaggle/working/" + NAME, "wb").write(data)
            rec["ok"] = True
            break
    except Exception as e:
        rec["tries"].append({{"url": url, "error": repr(e)[:300]}})
open("/kaggle/working/fetch.json", "w").write(json.dumps(rec, indent=1))
print(json.dumps(rec))
assert rec.get("ok"), "download or sha256 failed"
'''


def main():
    OUT.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# python-build-standalone 3.12.14 fetch (v13 input)\\n"]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": CODE.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    nb_bytes = json.dumps(nb, indent=1).encode()
    (OUT / f"{SLUG}.ipynb").write_bytes(nb_bytes)
    meta = {"id": f"f0909172434/{SLUG}", "title": SLUG.replace("-", " "), "code_file": f"{SLUG}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": True, "competition_sources": [], "dataset_sources": [],
            "kernel_sources": [], "model_sources": []}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"kernel": meta["id"], "notebook_sha256": hashlib.sha256(nb_bytes).hexdigest()}))


if __name__ == "__main__":
    main()
