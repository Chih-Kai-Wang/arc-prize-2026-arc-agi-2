"""Build the v13 Kaggle notebook: v11 NVARC pass A only (src byte-identical to v11; pass B disabled by the
ARC_V8_PASS_B_MIN_SECONDS env var) -> assemble (NVARC submission) -> v13 LLM phase (Qwen3.8-27B-FP8 program synthesis
from a Python 3.12 venv) -> finalize. Commit run = dev measurement (4-task NVARC smoke + keys30 program jobs, 120 min);
rerun = full pipeline. Never launches or submits.
  .venv/bin/python runs/arc-agi-2/v13/build.py arc-2026-v13-progsynth-v1
v3 (src3 + budget forcing; the flags go into the notebook's v13_llm.py call and preparation.json):
  .venv/bin/python runs/arc-agi-2/v13/build.py arc-2026-v13-progsynth-v3 --src src3 --llm-args '--force-at 58000 --answer-tokens 6000'
C2 (src6 = v2 + all-medium effort schedule + real-execution repair + full code; prereg-c2-effort-repair-v1.md):
  .venv/bin/python runs/arc-agi-2/v13/build.py arc-2026-v13-c2-medrep-v1 --src src6 \
      --llm-args '--effort-schedule medium:20000,medium:32000 --repair-rounds 2 --repair-cap 8000'
v5 (src5 = v2 + fresh program seeds + fixed concurrency, NVARC pass-A seeds offset; prereg-v5-reseed-c16-v1.md):
  .venv/bin/python runs/arc-agi-2/v13/build.py arc-2026-v13-progsynth-v5-rs16 --src src5 \
      --llm-args '--seed-tag v13s5 --conc-fixed 16' --nvarc-seed-offset 3000"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = HERE / "src"
V11_SRC = ROOT / "runs/arc-agi-2/v11/src"
sys.path.insert(0, str(HERE / "src7"))
import model_profiles as MP  # noqa: E402

NVARC_FILES = ("arc_v6_config.py", "arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py", "assemble.py")
V13_FILES = ("arc_llm.py", "progsynth.py", "v13_merge.py", "v13_llm.py", "model_profiles.py", "keys30.json",
             "qwen38_fp8_sha256.json")
FILES = NVARC_FILES + V13_FILES
NVARC_MODEL = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
QWEN38 = MP.PROFILES[MP.DEFAULT_PROFILE]["model_source"]
UNSLOTH = "sorokin/pip-install-unsloth-flash-patch"
WHEELS = "mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
PY312_KERNEL = "f0909172434/arc-2026-py312-fetch-v1"  # python-build-standalone 3.12.14 tarball (sha-checked)
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
COMPETITION = "arc-prize-2026-arc-agi-2"
COMMIT_SECONDS = 7200        # commit (dev) run: NVARC smoke + LLM phase end by this many seconds after global_start (v2: 64k jobs)
BOOT_OVERHEAD_SECONDS = 600  # docker pull/boot + mounts before the in-notebook clock starts
SRC_DIRS = ("src", "src3", "src5", "src6", "src7")  # src = v2 (frozen with its pushed build); src3 = v3 (budget forcing, off unless
#                                     flagged); src5 = v2 + --seed-tag / --conc-fixed (v5, only with V5_ARGS)
# the only flags a build may pass to v13_llm.py: v2 (none), v3 (force), v4 (cap, optionally with force)
LLM_ARGS = re.compile(r"(--cap (\d+)( --force-at (\d+) --answer-tokens \d+)?|--force-at (\d+) --answer-tokens \d+)?")
V5_ARGS = re.compile(r"--seed-tag (v13[a-z0-9]{1,8}) --conc-fixed (1[3-9]|2[0-4])")  # v5 / src5 only
# C2 / src6 only (prereg-c2-effort-repair-v1.md): all-medium effort schedule + real-execution repair; exact flags
C2_ARGS = re.compile(r"--effort-schedule medium:20000,medium:32000 --repair-rounds 2 --repair-cap 8000")
C2_COMMIT_SECONDS = 7000  # 13,800 GPU-s grant: v2/v3/v4 settled 13.75-13.87k at 7,200 (+600 boot) -> ~13.45k
# pass-A seed env vars of the v11 NVARC code (arc_solver.py defaults; same names as gcp-e1/driver_e1.py A_SEEDS)
NVARC_A_SEEDS = {"ARC_V8_TRAIN_AUG_SEED": 1, "ARC_V8_EVAL_AUG_SEED": 2, "ARC_V8_TRAINER_SEED": 42,
                 "ARC_V8_LORA_SEED": 42, "ARC_V8_SCORE_SEED_OFFSET": 0}


def nvarc_seed_env(offset):
    """Starter-cell env prefix for pass-A seeds + offset ('' for 0 = v2). E1 used offsets 0/1000/2000."""
    if offset not in (0, 3000, 4000, 5000):
        raise ValueError(f"--nvarc-seed-offset {offset} not allowed (0 = v2; 3000-5000 never used)")
    return "".join(f"{k}={v + offset} " for k, v in NVARC_A_SEEDS.items()) if offset else ""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def profile_files(model_profile=MP.DEFAULT_PROFILE):
    """Files bundled into the notebook for model_profile (default = FILES)."""
    if isinstance(model_profile, dict):
        profile = model_profile
    else:
        if model_profile not in MP.PROFILES:
            raise ValueError(f"unknown model profile: {model_profile!r}")
        if model_profile == MP.DEFAULT_PROFILE:
            return FILES
        profile = MP.PROFILES[model_profile]
    manifest = profile.get("sha_manifest")
    if (isinstance(manifest, str) and manifest and MP.TODO_FILL not in manifest
            and MP.is_launchable(profile)
            and manifest not in FILES):
        return FILES + (manifest,)
    return FILES


def llm_command(llm_args="", model_profile=MP.DEFAULT_PROFILE):
    """The notebook's v13_llm.py cell (launch_v13.gate checks the pushed cell against this)."""
    if model_profile not in MP.PROFILES:
        raise ValueError(f"unknown model profile: {model_profile!r}")
    prof_arg = f" --model-profile {model_profile}" if model_profile != MP.DEFAULT_PROFILE else ""
    if V5_ARGS.fullmatch(llm_args) or C2_ARGS.fullmatch(llm_args):
        return ("_comp_dir = cfg.competition_dir()\n"
                "!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {v13_end_time} --comp-dir {_comp_dir}"
                + prof_arg + " " + llm_args)
    m = LLM_ARGS.fullmatch(llm_args)
    if not m:
        raise ValueError(f"llm args not allowed: {llm_args!r}")
    cap = int(m.group(2)) if m.group(2) else 64000  # v13_llm.py default cap
    force = m.group(4) or m.group(5)
    if not 32000 <= cap <= 120000:  # MAX_LEN 131072 minus prompt + margin (v13_llm caps per job anyway)
        raise ValueError(f"--cap {cap} outside [32000, 120000]")
    if force and not 1000 <= int(force) <= cap - 1000:  # same bound v13_llm.py enforces at start
        raise ValueError(f"--force-at {force} outside [1000, cap - 1000]")
    return ("_comp_dir = cfg.competition_dir()\n"
            "!PYTHONUNBUFFERED=1 python v13_llm.py --end-time {v13_end_time} --comp-dir {_comp_dir}"
            + prof_arg + (" " + llm_args if llm_args else ""))


def notebook(sources, llm_args="", nvarc_seed_offset=0, commit_seconds=COMMIT_SECONDS,
             model_profile=MP.DEFAULT_PROFILE):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}}
    cells = [nbf.v4.new_markdown_cell(
        "# ARC-AGI-2 v13：NVARC 單輪 pass A＋Qwen3.8-27B 程式合成\n\n"
        "1. NVARC（v11 原碼不改；以環境變數關閉 pass B）跑完 pass A，組出 submission.json。\n"
        "2. 釋放 GPU 後，在獨立的 Python 3.12 venv 以 vLLM 0.19.0 服務 Qwen3.8-27B-FP8（TP4、MTP2），"
        "讓模型寫 `transform(grid)`；只有在增強後的全部訓練對上驗證通過的程式，其 test 答案才放進 attempt_1"
        "（attempt_2＝NVARC 第一名）。每個通過的程式都立即原子改寫 submission。\n"
        "3. 任一步失敗都保留 NVARC 的 submission。Commit 跑＝開發量測（4 題 NVARC 冒煙＋keys30 程式工作）。"),
        nbf.v4.new_code_cell("import time\nglobal_start = time.time()"),
        nbf.v4.new_code_cell("!pip uninstall -y tensorflow"),
        nbf.v4.new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv")]
    for name in profile_files(model_profile):
        cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + sources[name].decode()))
    cells.append(nbf.v4.new_code_cell(
        "import json\nimport arc_v6_config as cfg\n"
        "global_end_time = cfg.end_time(global_start)\n"
        f"v13_end_time = global_end_time if cfg.is_rerun() else global_start + {commit_seconds}\n"
        "with open(cfg.test_path()) as f:\n    _test = json.load(f)\n"
        "cfg.atomic_json(cfg.WORK_DIR + '/submission.json', cfg.fallback_submission(_test))\n"
        "cfg.log('fallback_written', rerun=cfg.is_rerun(), test_tasks=len(_test), model_dir=cfg.model_dir(),\n"
        "        competition_dir=cfg.competition_dir(),\n"
        "        work_file=cfg.work_challenges_path(), seconds_left=round(global_end_time - time.time()),\n"
        "        v13_seconds_left=round(v13_end_time - time.time()))"))
    cells.append(nbf.v4.new_code_cell(
        "!PYTHONUNBUFFERED=1 UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 "
        + nvarc_seed_env(nvarc_seed_offset) + "ARC_V8_PASS_B_MIN_SECONDS=1e12 python starter.py --end-time {global_end_time}"))
    cells.append(nbf.v4.new_code_cell("import assemble\nrecord = assemble.main()"))
    cells.append(nbf.v4.new_code_cell(llm_command(llm_args, model_profile)))
    cells.append(nbf.v4.new_code_cell(
        "import v13_merge\n"
        "if cfg.is_rerun():\n    v13_merge.finalize(cfg.WORK_DIR, cfg.test_path())\n"
        "with open(cfg.WORK_DIR + '/submission.json') as f:\n    cfg.validate_submission(_test, json.load(f))\n"
        "cfg.log('v13_done', seconds_left=round(global_end_time - time.time()))"))
    nb.cells = cells
    nbf.validate(nb)
    return nb


def build(slug, src="src", llm_args="", out=None, nvarc_seed_offset=0, model_profile=MP.DEFAULT_PROFILE):
    if not slug.startswith("arc-2026-v13-") or slug.startswith(("arc-2026-v13-e3-", "arc-2026-v13-e4-")):
        raise ValueError("slug must start with arc-2026-v13- (E3/E4 builds use build_e3.py/build_e4.py)")
    if src not in SRC_DIRS:
        raise ValueError(f"src must be one of {SRC_DIRS}")
    if model_profile not in MP.PROFILES:
        raise ValueError(f"unknown model profile: {model_profile!r}")
    profile = MP.PROFILES[model_profile]
    src_dir = HERE / src
    llm_command(llm_args, model_profile)  # validates the flags before anything is written
    nvarc_seed_env(nvarc_seed_offset)
    if (src == "src5") != bool(V5_ARGS.fullmatch(llm_args)) or (src == "src5") != bool(nvarc_seed_offset):
        raise ValueError("src5 goes with the v5 flags (--seed-tag/--conc-fixed) and a nonzero --nvarc-seed-offset, "
                         "and only with them")
    if (src == "src6") != bool(C2_ARGS.fullmatch(llm_args)) or (src == "src6" and nvarc_seed_offset):
        raise ValueError("src6 goes with the exact C2 flags and nvarc seed offset 0, and only with them")
    commit_seconds = C2_COMMIT_SECONDS if src == "src6" else COMMIT_SECONDS
    out = Path(out) if out else HERE / f"upload-{slug}"
    if out.exists():
        raise FileExistsError(f"{out} exists; builds are immutable")
    sources = {name: ((src_dir / name) if (src_dir / name).exists() else (HERE / "src7" / name)).read_bytes()
               for name in profile_files(model_profile)}
    for name in NVARC_FILES:  # NVARC stays exactly v11
        if sources[name] != (V11_SRC / name).read_bytes():
            raise ValueError(f"{name} differs from v11/src")
    for name, data in sources.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        else:
            json.loads(data)
    nb = notebook(sources, llm_args, nvarc_seed_offset, commit_seconds, model_profile)
    out.mkdir()
    code_file = slug + ".ipynb"
    nbf.write(nb, out / code_file)
    meta = {"id": "f0909172434/" + slug, "title": slug.replace("-", " "), "code_file": code_file,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": False, "competition_sources": [COMPETITION],
            "dataset_sources": [WHEELS], "kernel_sources": [UNSLOTH, PY312_KERNEL],
            "model_sources": [NVARC_MODEL, profile["model_source"]],
            "docker_image": IMAGE, "machine_shape": "NvidiaL4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    prep = {"created_at": datetime.now(timezone.utc).isoformat(), "kernel": meta["id"], "stage": "v13",
            "algorithm": "v11 NVARC pass A (pass B off) + Qwen3.8-27B-FP8 verified program synthesis (attempt_1)",
            "src_dir": str(src_dir.relative_to(ROOT)), "model_profile": model_profile,
            "llm_args": llm_args, "nvarc_seed_offset": nvarc_seed_offset,
            "source_sha256": {n: sha(d) for n, d in sources.items()},
            "notebook_sha256": sha((out / code_file).read_bytes()),
            "metadata_sha256": sha((out / "kernel-metadata.json").read_bytes()),
            "commit_wall_seconds": commit_seconds, "planned_commit_wall_seconds": commit_seconds + BOOT_OVERHEAD_SECONDS,
            "official_score": None}
    if src == "src6":
        prep.update(purpose="C2 measurement kernel: dev keys30 cell + one R1 exploratory submission",
                    approval="orchestrator 2026-10-19 reserve_grant_2026_10_19_arc_c2 (13,800 GPU-s; user 10/19 "
                             "'讓他們做'); prereg artifacts/verification/arc-agi-2/v13/prereg-c2-effort-repair-v1.md")
    (out / "preparation.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps({"folder": str(out), "kernel": meta["id"], "src_dir": prep["src_dir"],
                      "model_profile": model_profile, "llm_args": llm_args,
                      "notebook_sha256": prep["notebook_sha256"]}))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--src", default="src", choices=SRC_DIRS)
    ap.add_argument("--llm-args", default="")
    ap.add_argument("--nvarc-seed-offset", type=int, default=0)
    ap.add_argument("--model-profile", default=MP.DEFAULT_PROFILE, choices=tuple(MP.PROFILES))
    args = ap.parse_args()
    build(args.slug, args.src, args.llm_args, nvarc_seed_offset=args.nvarc_seed_offset,
          model_profile=args.model_profile)
