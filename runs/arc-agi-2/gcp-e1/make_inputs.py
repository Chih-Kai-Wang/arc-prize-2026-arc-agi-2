"""Build the E1 input bundle (gcp-1xl4-spec-v1): keys30, a test-challenges file holding those eval tasks, both srcs,
the driver and the unsloth patch cell, with a SHA256SUMS manifest. Upload the tgz to <your bucket>/arc/in/ (see startup.sh).
Usage: .venv/bin/python runs/arc-agi-2/gcp-e1/make_inputs.py
"""
import hashlib, io, json, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/arc-agi-2/official-v1"
sys.path.insert(0, str(ROOT / "runs/arc-agi-2/v11/src"))
import arc_v6_config as cfg  # noqa: E402


def keys30():
    ev = json.loads((DATA / "arc-agi_evaluation_challenges.json").read_text())
    ranked = sorted(ev, key=lambda k: (cfg.estimated_work(ev[k]), k))
    return ev, ranked[2::4]  # every 4th by size, offset 2 (prereg: stratified over the size range)


def main():
    ev, keys = keys30()
    files = {"in/keys30.json": json.dumps(keys).encode(),
             "in/arc-agi_test_challenges.json": json.dumps({k: ev[k] for k in keys}).encode()}
    for ver in ("v8", "v11"):
        for p in sorted((ROOT / f"runs/arc-agi-2/{ver}/src").glob("*.py")):
            files[f"in/{ver}/src/{p.name}"] = p.read_bytes()
    files["in/driver_e1.py"] = (HERE / "driver_e1.py").read_bytes()
    files["in/patch_cell.py"] = (ROOT / "runs/arc-agi-2/colab-cal-v1/patch_cell.py").read_bytes()
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n[3:]}\n" for n, b in sorted(files.items()))
    files["in/SHA256SUMS"] = sums.encode()
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    tgz = build / "e1-inputs-v1.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = {"tgz": str(tgz.relative_to(ROOT)), "tgz_sha256": hashlib.sha256(tgz.read_bytes()).hexdigest(),
                "keys30": keys, "outputs": sum(len(ev[k]["test"]) for k in keys), "files": sums.splitlines()}
    (build / "e1-inputs-v1.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("tgz", "tgz_sha256", "outputs")}))


if __name__ == "__main__":
    main()
