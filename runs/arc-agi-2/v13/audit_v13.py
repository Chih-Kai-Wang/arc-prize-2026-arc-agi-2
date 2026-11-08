"""Audit a downloaded v13 commit (dev) run against the pre-registered rules
(artifacts/verification/arc-agi-2/v13/prereg-dev-run-v1.md: integration pass/fail + ship rule). Writes
artifacts/verification/arc-agi-2/v6/audit-SLUG.json; cloud.py submit SLUG --audit <that file> reads submission_allowed.

  .venv/bin/python runs/arc-agi-2/v13/audit_v13.py hidden-shape SLUG      # runs tests(3)/test_v13.py, records hashes
  .venv/bin/python runs/arc-agi-2/v13/audit_v13.py audit SLUG --hidden-shape-test H.json --review R.json
v3 builds (src3 with --force-at) also need the HA replay of their programs (prereg "v3 預登記" items 1-4):
  .venv/bin/python runs/arc-agi-2/v13/audit_v13.py audit SLUG --hidden-shape-test H.json --review R.json --replay P.json

Reads runs/arc-agi-2/v13/upload-SLUG (the pushed build) and runs/arc-agi-2/v6/output-SLUG (cloud.py download).
Eval correctness is the pre-registered ship signal (keys30 = public eval, measured with augmentation; pretraining
contamination cannot be excluded). submission_allowed is true only when every gate passes.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT_ROOT = ROOT / "runs/arc-agi-2/v6"
E = ROOT / "artifacts/verification/arc-agi-2/v6"
HS_DIR = ROOT / "artifacts/verification/arc-agi-2/v13"
DATA = ROOT / "data/arc-agi-2/official-v1"
BUILD = HERE / "build.py"
V2_REPLAY = HS_DIR / "replay-v2-merge.json"  # v2's HA replay (F2): the bar a v3 build must reach (mean 8.778)
# pinned (T3 review 10-16, Opus-1): replay_v2_merge.py's docstring writes to the same path, so an overwrite with a
# v3 replay would compare v3 with itself
V2_REPLAY_SHA256 = "3676258e78143497c50604796e67295e4bccfdb519e62d448f33da7207534619"


def test_cmd(src_dir):
    """The local suite that covers a build's sources: tests/ for src (v2), tests3/ for src3 (v3), tests5/ for src5 (v5)."""
    tests = {"runs/arc-agi-2/v13/src": "tests", "runs/arc-agi-2/v13/src3": "tests3",
             "runs/arc-agi-2/v13/src5": "tests5", "runs/arc-agi-2/v13/src6": "tests6"}[str(src_dir)]
    return ["uv", "run", "--quiet", "--no-project", "--with", "numpy", "--with", "requests", "python",
            f"runs/arc-agi-2/v13/{tests}/test_v13.py"]


TEST_CMD = test_cmd("runs/arc-agi-2/v13/src")
sys.path.insert(0, str(HERE / "src"))
import v13_merge as M  # noqa: E402  (pure: validate / valid_grid)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_constants(path=BUILD):
    """String/tuple constants of build.py, read without importing it (nbformat is not needed here)."""
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def log_text(out):
    parts = []
    for log in sorted(Path(out).glob("*.log")):
        raw = log.read_text(errors="ignore")
        try:
            parts.append("".join(item.get("data", "") for item in json.loads(raw)))
        except (ValueError, AttributeError):
            parts.append(raw)
    return "\n".join(parts)


def read_rows(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def valid(test, submission):
    try:
        return M.validate(test, submission) is True
    except Exception:  # noqa: BLE001
        return False


def ship_rule(rows, keys30):
    """Pre-registered: >= 1 keys30 row verified and correct (every test output), and precision not catastrophic =
    (verified < 3) or (correct / verified >= 0.5). Counted per job row, as the driver's summary does."""
    ver = [r for r in rows if r.get("verified")]
    correct = [r for r in ver if r.get("correct") and all(r["correct"])]
    k_correct = [r for r in correct if r["task"] in keys30]
    precision_ok = len(ver) < 3 or len(correct) / len(ver) >= 0.5
    facts = {"verified_rows": len(ver), "verified_correct_rows": len(correct), "keys30_verified_correct_rows": len(k_correct),
             "verified_tasks": sorted({r["task"] for r in ver}), "correct_tasks": sorted({r["task"] for r in correct}),
             "keys30_correct_tasks": sorted({r["task"] for r in k_correct}),
             "false_positive_tasks": sorted({r["task"] for r in ver} - {r["task"] for r in correct}),
             "correctness_measured": all("correct" in r for r in ver),
             "precision": round(len(correct) / len(ver), 3) if ver else None}
    return {"keys30_verified_correct": bool(k_correct) and facts["correctness_measured"],
            "precision_floor": precision_ok}, facts


def replay_mean(path):
    seeds = json.loads(Path(path).read_text())["seeds"]
    return sum(v["after"] for v in seeds.values()) / len(seeds)


def v3_gates(llm, rows, replay, replay_excl=None):
    """Pre-registered v3 additions (prereg-dev-run-v1.md "v3 預登記" items 1-4 + its pre-push amendment): forcing
    really active (parity ok, never disabled), >= 3 forcings whose continuation really ran (done, > 0 tokens, no
    force_error), the forced-source precision floor, the HA replay of this run's programs (replay_v2_merge.py) at least
    v2's mean, and >= 1 forced-source verified-correct task that raised the HA replay score (forcing contributed)."""
    forced = [r for r in rows if r.get("forced") and not r.get("force_error") and r.get("cont_state") == "done"
              and (r.get("cont_completion_tokens") or 0) > 0]
    fv = [r for r in rows if r.get("verified") and r.get("source") == "forced"]
    fc = [r for r in fv if r.get("correct") and all(r["correct"])]
    gates = {"v3_forcing_active": bool(llm.get("force_at")) and (llm.get("token_id_parity") or {}).get("ok") is True
             and "force_disabled" not in llm,
             "v3_forced_executed_ge3": len(forced) >= 3,
             "v3_forced_precision_floor": len(fv) < 3 or len(fc) / len(fv) >= 0.5}
    facts = {"forced_executed": len(forced), "forced_verified": len(fv), "forced_correct": len(fc),
             "v2_replay_mean": round(replay_mean(V2_REPLAY), 3)}
    gates["v2_baseline_pinned"] = hashlib.sha256(V2_REPLAY.read_bytes()).hexdigest() == V2_REPLAY_SHA256
    try:
        rp = json.loads(Path(replay).read_text())
        program_tasks = {r["task"] for r in rows if r.get("verified")} & set(json.loads(
            (HERE / "src3/keys30.json").read_text()))
        facts.update(replay=str(replay), replay_mean=round(replay_mean(replay), 3), replay_rep=rp.get("rep"))
        gates["v3_replay_mean_ge_v2"] = (set(rp["program_tasks"]) == program_tasks and rp.get("rep") is None
                                         and rp.get("exclude_source") is None and gates["v2_baseline_pinned"]
                                         and replay_mean(replay) >= replay_mean(V2_REPLAY) - 1e-9)
        # T3 review 10-16 (Argon-1 / Opus-3): forcing's contribution = full replay minus the paired
        # --exclude-source forced replay (not the full replay's per-task delta, which also credits natural stops);
        # forcing may not be net-negative, and no displacement may lose a correct NVARC answer (v2's F2 rule)
        rx = json.loads(Path(replay_excl).read_text())
        need_ok = rx.get("exclude_source") == "forced" and rx.get("rep") is None and len(rx["seeds"]) == len(rp["seeds"])
        mean = lambda d, t: sum(v["per_task_delta"].get(t, 0) for v in d["seeds"].values()) / len(d["seeds"])
        gain = {t: mean(rp, t) - mean(rx, t) for t in {r["task"] for r in fc}}
        facts.update(forced_correct_task_ha_gain=gain, replay_excl=str(replay_excl),
                     forcing_paired_delta=round(replay_mean(replay) - replay_mean(replay_excl), 3),
                     lost_correct_nvarc_any=rp.get("lost_correct_nvarc_any"))
        gates["v3_forced_contributed"] = need_ok and any(g > 0 for g in gain.values())
        gates["v3_forcing_not_negative"] = need_ok and replay_mean(replay) >= replay_mean(replay_excl) - 1e-9
        gates["v3_no_lost_correct_nvarc"] = rp.get("lost_correct_nvarc_any") is False
    except Exception as exc:  # noqa: BLE001  (missing / wrong replay file: the gates fail)
        facts["replay_error"] = repr(exc)[:300]
        for k in ("v3_replay_mean_ge_v2", "v3_forced_contributed", "v3_forcing_not_negative", "v3_no_lost_correct_nvarc"):
            gates[k] = False
    return gates, facts


def v4_cap_applied(llm, rows, want):
    """prereg-v4-v1.md section 3 item 2: the dev run really used the requested cap (summary and every row)."""
    return llm.get("cap") == want and all(int(r.get("cap") or 0) <= want for r in rows)


def v4_gates(llm, rows, slug, replay, replay_excl=None, want_force=0):
    """prereg-v4-v1.md section 3 (T3 review 10-16): a v4 dev run (7,200 s commit, cap 80k/96k) finishes about one wave
    and only ~6-7 keys30 tasks, so v3's measurement gates (>= 3 forcings, replay >= v2's mean, forcing contributed)
    are not measurable and do not apply. v4 instead needs: the HA replay of its own raw v13-programs.jsonl (file named
    replay-SLUG-full.json) not below HA on any seed and losing no correct NVARC answer (= no harm vs NVARC, v2's F2
    rule; NOT a no-regression check against v2, whose gap is only recorded); with forcing requested (c80f only): the
    requested force point really active (parity ok, never disabled), the forced-source precision floor, and forcing
    not net-negative against the paired replay-SLUG-excl-forced.json."""
    gates, facts = {}, {}
    fv = [r for r in rows if r.get("verified") and r.get("source") == "forced"]
    fc = [r for r in fv if r.get("correct") and all(r["correct"])]
    if want_force:
        gates["v4_forcing_active"] = (llm.get("force_at") == want_force and "force_disabled" not in llm
                                      and (llm.get("token_id_parity") or {}).get("ok") is True)
        gates["v4_forced_precision_floor"] = len(fv) < 3 or len(fc) / len(fv) >= 0.5
        executed = [r for r in rows if r.get("forced") and not r.get("force_error") and r.get("cont_state") == "done"]
        gates["v4_forcing_executed"] = len(executed) >= 1  # delta review 10-16 P2: c80f must not ship on 0 real forcings
        facts.update(forced_verified=len(fv), forced_correct=len(fc), forced_executed=len(executed))
    try:
        rp = json.loads(Path(replay).read_text())
        program_tasks = {r["task"] for r in rows if r.get("verified")} & set(json.loads(
            (HERE / "src3/keys30.json").read_text()))
        deltas = [v["delta"] for v in rp["seeds"].values()]
        facts.update(replay=str(replay), replay_deltas=deltas, replay_mean=round(replay_mean(replay), 3),
                     lost_correct_nvarc_any=rp.get("lost_correct_nvarc_any"))
        if V2_REPLAY.exists() and hashlib.sha256(V2_REPLAY.read_bytes()).hexdigest() == V2_REPLAY_SHA256:  # record only
            facts["replay_mean_minus_v2"] = round(replay_mean(replay) - replay_mean(V2_REPLAY), 3)
        gates["v4_replay_no_harm_vs_ha"] = (Path(replay).name == f"replay-{slug}-full.json"
                                            and set(rp["program_tasks"]) == program_tasks and rp.get("rep") is None
                                            and rp.get("exclude_source") is None and len(deltas) == 3
                                            and all(x >= -1e-9 for x in deltas))
        gates["v4_no_lost_correct_nvarc"] = rp.get("lost_correct_nvarc_any") is False
        if want_force:
            rx = json.loads(Path(replay_excl).read_text())
            unforced = {r["task"] for r in rows if r.get("verified") and r.get("source") != "forced"}
            gates["v4_forcing_not_negative"] = (Path(replay_excl).name == f"replay-{slug}-excl-forced.json"
                                                and set(rx["program_tasks"]) == unforced & set(json.loads(
                                                    (HERE / "src3/keys30.json").read_text()))
                                                and rx.get("exclude_source") == "forced" and rx.get("rep") is None
                                                and len(rx["seeds"]) == len(rp["seeds"])
                                                and replay_mean(replay) >= replay_mean(replay_excl) - 1e-9)
            facts["forcing_paired_delta"] = round(replay_mean(replay) - replay_mean(replay_excl), 3)
    except Exception as exc:  # noqa: BLE001  (missing / wrong replay file: the gates fail)
        facts["replay_error"] = repr(exc)[:300]
        for k in ("v4_replay_no_harm_vs_ha", "v4_no_lost_correct_nvarc") + (("v4_forcing_not_negative",) if want_force else ()):
            gates[k] = False
    return gates, facts


V5_ARGS = re.compile(r"--seed-tag (v13[a-z0-9]{1,8}) --conc-fixed (\d+)")
C2_ARGS = re.compile(r"--effort-schedule medium:20000,medium:32000 --repair-rounds 2 --repair-cap 8000")  # = build.py
V2_DEV_JOBS, V2_DEV_TOKENS = 26, 1168567  # v2 dev run (v2-devrun-timeline-v1.json): same 7,200 s commit window


def v5_gates(llm, out, llm_args):
    """prereg-v5-reseed-c16-v1.md B + C: the flags really applied, no KV preemption, and throughput not below v2."""
    m = V5_ARGS.fullmatch(llm_args)
    tag, conc = m.group(1), int(m.group(2))
    metrics = read_rows(out / "v13-metrics.jsonl")
    pre = [r["vllm:num_preemptions_total"] for r in metrics if isinstance(r.get("vllm:num_preemptions_total"), (int, float))]
    kv = [r["vllm:kv_cache_usage_perc"] for r in metrics if isinstance(r.get("vllm:kv_cache_usage_perc"), (int, float))]
    summ = llm.get("summary") or {}
    gates = {"v5_seed_tag_applied": llm.get("seed_tag") == tag and llm.get("version") == "v13-llm-v5",
             "v5_conc_fixed_applied": llm.get("conc_fixed") == conc and llm.get("conc_effective") == conc,
             "v5_no_preemption": bool(pre) and max(pre) == 0,
             "v5_kv_peak_le_095": bool(kv) and max(kv) <= 0.95,
             "v5_jobs_ge_v2": (summ.get("jobs") or 0) >= V2_DEV_JOBS,
             "v5_tokens_ge_095_v2": (summ.get("tokens") or 0) >= 0.95 * V2_DEV_TOKENS}
    facts = {"seed_tag": llm.get("seed_tag"), "conc_effective": llm.get("conc_effective"), "jobs": summ.get("jobs"),
             "tokens": summ.get("tokens"), "throughput_ratio_vs_v2": round((summ.get("tokens") or 0) / V2_DEV_TOKENS, 3),
             "preemptions_max": max(pre) if pre else None, "kv_peak": round(max(kv), 3) if kv else None}
    return gates, facts


def c2_gates(llm, out, rows):
    """prereg-c2-effort-repair-v1.md section 3 (+ amendment 1): schedule/effort really applied, the repair loop fired,
    full code stored, no KV preemption, KV peak <= 0.95, no job exceptions."""
    metrics = read_rows(out / "v13-metrics.jsonl")
    pre = [r["vllm:num_preemptions_total"] for r in metrics if isinstance(r.get("vllm:num_preemptions_total"), (int, float))]
    kv = [r["vllm:kv_cache_usage_perc"] for r in metrics if isinstance(r.get("vllm:kv_cache_usage_perc"), (int, float))]
    jobs = [r for r in rows if r.get("state") not in (None, "prompt_too_long", "job_exception")]
    rep0 = sorted(r.get("completion_tokens") or 0 for r in jobs if r.get("rep") == 0 and r.get("state") == "done")
    ver = [r for r in rows if r.get("verified")]
    gates = {"c2_schedule_applied": llm.get("effort_schedule") == [["medium", 20000], ["medium", 32000]]
             and llm.get("repair_rounds") == 2 and llm.get("repair_cap") == 8000 and bool(llm.get("effort_smoke")),
             "c2_effort_medium_all_jobs": bool(jobs) and all(r.get("effort") == "medium" for r in jobs),
             # prereg amendment 7 (Argon pre-push P0): caps enforced and the schedule ran; the rep-0 median is a fact
             "c2_rep0_cap_enforced": bool(rep0) and max(rep0) <= 20000 + 8,  # amendment 8: MTP accounting slack
             "c2_rep_ge1_ran": any((r.get("rep") or 0) >= 1 for r in jobs),
             "c2_repair_fired": any(r.get("repair") for r in rows),
             "c2_full_code_stored": all(r.get("code") for r in ver)
             and all(r.get("best_code") is not None for r in jobs if not r.get("verified") and r.get("tried")),
             "c2_no_preemption": bool(pre) and max(pre) == 0,
             "c2_kv_peak_le_095": bool(kv) and max(kv) <= 0.95,
             "c2_no_job_exceptions": not any(r.get("state") == "job_exception" for r in rows)}
    facts = {"jobs": len(jobs), "rep0_done": len(rep0), "rep0_median_tokens": rep0[len(rep0) // 2] if rep0 else None,
             "repaired": sum(str(r.get("source", "")).startswith("repair") for r in ver),
             "repair_jobs": sum(bool(r.get("repair")) for r in rows), "conc_effective": llm.get("conc_effective"),
             "preemptions_max": max(pre) if pre else None, "kv_peak": round(max(kv), 3) if kv else None,
             "effort_smoke": llm.get("effort_smoke")}
    return gates, facts


def review_go(review, legacy):
    """Independent-review gate. v2 (legacy) kept its free-text verdict. v3/v4 (T3 review 10-16, advisor ruling): the
    JSON lists >= 2 raw reviews whose files exist with matching sha256, none NO-GO, and every P0/P1 of either review is
    REFUTED or FIXED_OUTSIDE_KERNEL with evidence; any finding needing a change inside the pushed kernel blocks."""
    r = json.loads(Path(review).read_text())
    if legacy:
        return r.get("verdict") in ("GO", "GO-with-fixes-applied")
    revs = r.get("reviews") or []
    files_ok = len(revs) >= 2 and all(Path(ROOT / x["file"]).exists()
                                      and hashlib.sha256((ROOT / x["file"]).read_bytes()).hexdigest() == x["sha256"]
                                      for x in revs)
    verdicts_ok = all(str(x.get("verdict", "")).upper() in ("GO", "GO-WITH-FIXES") for x in revs)
    findings_ok = all(f.get("status") in ("REFUTED", "FIXED_OUTSIDE_KERNEL") and f.get("evidence")
                      for f in r.get("p0p1", []))
    return (r.get("verdict") in ("GO", "GO-with-fixes-applied") and files_ok and verdicts_ok and findings_ok
            and r.get("kernel_code_change_required") is False)


def audit(slug, hidden_shape, review, upload=None, out=None, e_dir=E, data=DATA, build=BUILD, replay=None,
          replay_excl=None):
    upload = Path(upload) if upload else HERE / f"upload-{slug}"
    out = Path(out) if out else OUTPUT_ROOT / f"output-{slug}"
    gates, facts = {}, {}
    # 1. the pushed build is what build.py produced from the current src
    prep = json.loads((upload / "preparation.json").read_text())
    if prep.get("stage") in ("v13-e3", "v13-e4", "v13-e5") or slug.startswith(("arc-2026-v13-e3-", "arc-2026-v13-e4-",
                                                                                 "arc-2026-v13-e5-")):
        raise ValueError("E3/E4 are offline measurement kernels, never submission candidates: no audit")
    meta = json.loads((upload / "kernel-metadata.json").read_text())
    nb_sha = sha256_file(upload / meta["code_file"])
    src_dir = ROOT / prep["src_dir"]
    src_now = {n: sha256_file(src_dir / n if (src_dir / n).exists() else HERE / "src7" / n) for n in prep["source_sha256"]}  # model_profiles.py lives only in src7 (launch_v13.gate does the same)
    gates["build_hashes_match"] = (meta["id"] == prep["kernel"] == "f0909172434/" + slug and nb_sha == prep["notebook_sha256"]
                                   and sha256_file(upload / "kernel-metadata.json") == prep["metadata_sha256"]
                                   and src_now == prep["source_sha256"])
    const = build_constants(build)
    if const.get("QWEN38") is None:  # since 10-24 build.py reads it from model_profiles (not a literal); use the copy
        mp_path = src_dir / "model_profiles.py"  # whose sha256 the build recorded (src5 has none on disk -> src7)
        mp_path = mp_path if mp_path.exists() else HERE / "src7" / "model_profiles.py"
        if sha256_file(mp_path) == prep["source_sha256"].get("model_profiles.py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_profiles_build", mp_path)
            MP = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(MP)
            const["QWEN38"] = MP.PROFILES[prep.get("model_profile") or MP.DEFAULT_PROFILE]["model_source"]
    gates["metadata_private_offline_l4"] = (meta.get("is_private") is True and meta.get("enable_internet") is False
                                            and meta.get("enable_gpu") is True and meta.get("machine_shape") == "NvidiaL4")
    gates["metadata_sources_match_build"] = (meta.get("competition_sources") == [const.get("COMPETITION")]
                                             and meta.get("dataset_sources") == [const.get("WHEELS")]
                                             and meta.get("model_sources") == [const.get("NVARC_MODEL"), const.get("QWEN38")]
                                             and meta.get("kernel_sources") == [const.get("UNSLOTH"), const.get("PY312_KERNEL")]
                                             and meta.get("docker_image") == const.get("IMAGE"))
    facts.update(notebook_sha256=nb_sha, src_dir=prep["src_dir"], commit_wall_seconds=prep.get("commit_wall_seconds"))
    # 2. integration (pre-registered items 1-5)
    text = log_text(out)
    assembly = json.loads((out / "v6-assembly.json").read_text()) if (out / "v6-assembly.json").exists() else {}
    starts = sorted(set(int(r) for r in re.findall(r"\[Rank (\d)\] start!", text)))
    exits = re.findall(r'"workers_exited", "exitcodes": (\[[^\]]*\])', text)
    failed = re.findall(r"\[Rank (\d)\] task (\S+) failed: (.*)", text)
    smoke = re.findall(r'"eval_smoke_score", "solved": ([\d.]+), "tasks": (\d+)', text)
    gates["nvarc_assembly_no_error"] = bool(assembly) and assembly.get("error") is None
    gates["nvarc_four_workers_clean"] = starts == [0, 1, 2, 3] and bool(exits) and all(
        json.loads(e) == [0, 0, 0, 0] for e in exits) and not failed
    llm = json.loads((out / "v13-llm.json").read_text()) if (out / "v13-llm.json").exists() else {}
    ph = llm.get("phases", {})
    chosen = (ph.get("server") or {}).get("chosen") or {}
    selftest = llm.get("sandbox_selftest") or [None]
    gates["llm_not_skipped"] = bool(llm) and llm.get("skipped") is None
    gates["llm_setup_ok"] = ((ph.get("gpus") or {}).get("free_after") is not None and (ph.get("install") or {}).get("pip_rc") == 0
                             and not (ph.get("wheels") or {}).get("mismatched", ["?"]) and chosen.get("smoke_ok") is True
                             and isinstance(selftest[0], list) and bool(selftest[0])
                             and all(isinstance(x, dict) and "ok" in x for x in selftest[0]))
    rows = read_rows(out / "v13-programs.jsonl")
    finished = [r for r in rows if r.get("finish_reason") or r.get("state") == "aborted"]
    gates["llm_jobs_recorded"] = bool(finished)
    gates["llm_returned_before_end"] = bool(llm) and llm.get("t0", 0) + llm.get("elapsed", 1e18) <= llm.get("end_time", 0)
    gates["notebook_v13_done"] = "v13_done" in text and not re.search(r"Traceback \(most recent call last\)", text)
    test = json.loads((data / "arc-agi_test_challenges.json").read_text())
    try:
        gates["test_submission_valid"] = valid(test, json.loads((out / "submission.json").read_text()))
    except Exception as exc:  # noqa: BLE001
        gates["test_submission_valid"] = False
        facts["submission_error"] = repr(exc)[:300]
    if any(r.get("verified") for r in rows):
        ev = json.loads((data / "arc-agi_evaluation_challenges.json").read_text())
        try:
            gates["eval_v13_submission_valid"] = valid(ev, json.loads((out / "eval-v13-submission.json").read_text()))
        except Exception as exc:  # noqa: BLE001
            gates["eval_v13_submission_valid"] = False
            facts["eval_v13_submission_error"] = repr(exc)[:300]
    # 3. ship rule (pre-registered)
    keys30 = json.loads((src_dir / "keys30.json").read_text())
    ship, ship_facts = ship_rule(rows, set(keys30))
    gates.update(ship)
    facts["ship_rule"] = ship_facts
    if "--cap" in prep.get("llm_args", ""):  # v4: the dev run really used the requested cap (E4 -> prereg v4 rule)
        want = int(prep["llm_args"].split("--cap ", 1)[1].split()[0])
        gates["v4_cap_applied"] = v4_cap_applied(llm, rows, want)
        facts["v4_cap"] = {"requested": want, "summary_cap": llm.get("cap")}
        force = int(prep["llm_args"].split("--force-at ", 1)[1].split()[0]) if "--force-at" in prep["llm_args"] else 0
        g4, f4 = v4_gates(llm, rows, slug, replay, replay_excl, force)
        gates.update(g4)
        facts["v4"] = f4
    elif "--force-at" in prep.get("llm_args", ""):
        g3, f3 = v3_gates(llm, rows, replay, replay_excl)
        gates.update(g3)
        facts["v3"] = f3
    elif V5_ARGS.fullmatch(prep.get("llm_args", "")):
        g5, f5 = v5_gates(llm, out, prep["llm_args"])
        gates.update(g5)
        facts["v5"] = f5
    elif C2_ARGS.fullmatch(prep.get("llm_args", "")):
        g6, f6 = c2_gates(llm, out, rows)
        gates.update(g6)
        facts["c2"] = f6
    # measurements (recorded, never gated; model hash is record-only by pre-registration)
    facts.update(summary=llm.get("summary"), cap=llm.get("cap"), conc_effective=llm.get("conc_effective"),
                 effort=llm.get("effort"), gen_seconds=llm.get("gen_seconds"), aborted_streams=llm.get("aborted_streams"),
                 server=dict((k, chosen.get(k)) for k in ("mtp", "eager", "startup_seconds", "kv_tokens")),
                 model_hash=ph.get("model_hash"), order=llm.get("order"),
                 rejected_by_guards=sorted({(r["task"], t["rejected"]) for r in rows for t in r.get("tried", [])
                                            if t.get("rejected")}),
                 nvarc_eval_smoke=({"solved": float(smoke[-1][0]), "tasks": int(smoke[-1][1])} if smoke else None),
                 changed_outputs=llm.get("changed_outputs"))
    # 4. hidden-shape test of exactly this build, and an independent review
    try:
        hs = json.loads(Path(hidden_shape).read_text())
        gates["hidden_shape_test_passed"] = (hs.get("passed") is True and hs.get("source_sha256") == prep["source_sha256"]
                                             and hs.get("notebook_sha256") == nb_sha
                                             and hs.get("command") == " ".join(test_cmd(prep["src_dir"])))
    except Exception:  # noqa: BLE001
        gates["hidden_shape_test_passed"] = False
    if C2_ARGS.fullmatch(prep.get("llm_args", "")):  # < 15,000 GPU-s exploratory: R1 automated gates only
        facts["review"] = "not required: submission-authority-v1.json strategy_amendment_2026_10_19 R1"
    else:
        try:
            gates["independent_review_go"] = review_go(review, legacy=not prep.get("llm_args"))
        except Exception:  # noqa: BLE001
            gates["independent_review_go"] = False

    def rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except (ValueError, TypeError):
            return str(p)

    result = {"kernel": "f0909172434/" + slug, "at": datetime.now(timezone.utc).isoformat(), "gates": gates, "facts": facts,
              "hidden_shape_test": rel(hidden_shape), "review": rel(review),
              "prereg": ("artifacts/verification/arc-agi-2/v13/prereg-v5-reseed-c16-v1.md"
                         if V5_ARGS.fullmatch(prep.get("llm_args", "")) else
                         "artifacts/verification/arc-agi-2/v13/prereg-c2-effort-repair-v1.md"
                         if C2_ARGS.fullmatch(prep.get("llm_args", "")) else
                         "artifacts/verification/arc-agi-2/v13/prereg-dev-run-v1.md"),
              "submission_allowed": all(gates.values()), "official_score": None}
    path = Path(e_dir) / f"audit-{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=list) + "\n")
    print(json.dumps({"submission_allowed": result["submission_allowed"],
                      "failed_gates": [k for k, v in gates.items() if not v], "ship_rule": ship_facts,
                      "summary": facts["summary"], "audit": rel(path)}, ensure_ascii=False))
    return result


def hidden_shape(slug, upload=None, out_dir=HS_DIR):
    """Run the v13 local test suite (hidden-shape rerun, commit path, skips, abort salvage, guards) on the current src
    and record it with the hashes the audit checks."""
    upload = Path(upload) if upload else HERE / f"upload-{slug}"
    prep = json.loads((upload / "preparation.json").read_text())
    meta = json.loads((upload / "kernel-metadata.json").read_text())
    src_dir = ROOT / prep["src_dir"]
    src_now = {n: sha256_file(src_dir / n if (src_dir / n).exists() else HERE / "src7" / n) for n in prep["source_sha256"]}  # model_profiles.py lives only in src7 (launch_v13.gate does the same)
    cmd = test_cmd(prep["src_dir"])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    record = {"at": datetime.now(timezone.utc).isoformat(), "kernel": prep["kernel"], "command": " ".join(cmd),
              "returncode": p.returncode, "all_ok_printed": "ALL OK" in p.stdout, "stdout_tail": p.stdout[-3000:],
              "stderr_tail": p.stderr[-3000:], "source_sha256": src_now,
              "notebook_sha256": sha256_file(upload / meta["code_file"])}
    record["passed"] = p.returncode == 0 and record["all_ok_printed"] and src_now == prep["source_sha256"]
    path = Path(out_dir) / f"hidden-shape-{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"passed": record["passed"], "returncode": p.returncode, "file": str(path)}))
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hidden-shape")
    h.add_argument("slug")
    h.add_argument("--out-dir", default=str(HS_DIR))
    a = sub.add_parser("audit")
    a.add_argument("slug")
    a.add_argument("--hidden-shape-test", required=True)
    a.add_argument("--review", required=True)
    a.add_argument("--e-dir", default=str(E))
    a.add_argument("--replay", default=None)  # v3: replay_v2_merge.py output for this run's programs
    a.add_argument("--replay-excl", default=None)  # v3: the same replay with --exclude-source forced (paired)
    args = ap.parse_args()
    if args.cmd == "hidden-shape":
        hidden_shape(args.slug, out_dir=Path(args.out_dir))
    else:
        audit(args.slug, args.hidden_shape_test, args.review, e_dir=Path(args.e_dir), replay=args.replay,
              replay_excl=args.replay_excl)
