"""Unit test for runs/arc-agi-2/fork-perfpatch/audit_fork.py against a small fake output directory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_fork", ROOT / "runs/arc-agi-2/fork-perfpatch/audit_fork.py")
audit_fork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_fork)

CHALLENGES = {"aaaa0001": {"test": [{}]}, "bbbb0002": {"test": [{}, {}]}, **{t: {"test": [{}]} for t in audit_fork.SMOKE_IDS}}
G = [[1, 2], [3, 4]]
GOOD_LOG = "".join(f"[Rank {r}] start!\n[Rank {r}] allocated 100MB for training\n[Rank {r}] finished {t} in {50 + r}.0s\n"
                   for r, t in enumerate(audit_fork.SMOKE_IDS)) + "*** Reload score: 1.5\n"
BUILD_PY = 'MODEL = "m/1"\nUNSLOTH = "u/k"\nIMAGE = "img@sha"\n'
META = {"code_file": "n.ipynb", "enable_internet": False, "is_private": True, "machine_shape": "NvidiaL4",
        "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["m/1"], "kernel_sources": ["u/k"],
        "dataset_sources": [], "docker_image": "img@sha"}


class AuditFork(unittest.TestCase):
    def build(self, tmp, log=GOOD_LOG, submission="good", passed=True, verdict="GO", stream=True, meta=None,
              hs_extra=None, ext_bytes=b"nb", smoke_blank=None, second_log=None, decoded=None):
        tmp = Path(tmp)
        out = tmp / "out"
        out.mkdir()
        entries = {t: [{"attempt_1": G, "attempt_2": [[0]]} for _ in v["test"]] for t, v in CHALLENGES.items()}
        if smoke_blank:
            entries[smoke_blank] = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
        if submission == "missing":
            del entries["aaaa0001"]
        elif submission == "badgrid":
            entries["aaaa0001"][0]["attempt_1"] = [[10]]
        elif submission == "short":
            entries["bbbb0002"] = entries["bbbb0002"][:1]
        elif submission == "extra":
            entries["zzzz9999"] = [{"attempt_1": G, "attempt_2": G}]
        if submission != "none":
            (out / "submission.json").write_text(json.dumps(entries))
        (out / "x.log").write_text(json.dumps([{"stream_name": "stdout", "data": log}]) if stream else log)
        if second_log is not None:
            (out / "y.log").write_text(json.dumps([{"stream_name": "stdout", "data": second_log}]))
        if decoded is not None:
            (out / "inference_outputs").mkdir()
            for name in decoded:
                (out / "inference_outputs" / name).write_bytes(b"x")
        (tmp / "challenges.json").write_text(json.dumps(CHALLENGES))
        up = tmp / "up"
        up.mkdir()
        (up / "n.ipynb").write_bytes(b"nb")
        (tmp / "ext.ipynb").write_bytes(ext_bytes)
        (tmp / "build.py").write_text(BUILD_PY)
        src = tmp / "src"
        src.mkdir()
        (src / "a.py").write_text("print(1)\n")
        sha = hashlib.sha256(b"nb").hexdigest()
        (up / "kernel-metadata.json").write_text(json.dumps(meta or META))
        (up / "preparation.json").write_text(json.dumps({"source_notebook": "a/b", "source_notebook_sha256": sha, "notebook_sha256": sha}))
        evidence = {"passed": passed, "src_sha256": {"a.py": hashlib.sha256(b"print(1)\n").hexdigest()}, "notebook_sha256": sha}
        evidence.update(hs_extra or {})
        (tmp / "hs.json").write_text(json.dumps(evidence))
        (tmp / "rv.json").write_text(json.dumps({"verdict": verdict}))
        return dict(hidden_shape=tmp / "hs.json", review=tmp / "rv.json", out=out, eval_path=tmp / "challenges.json",
                    upload=up, e_dir=tmp / "e", pinned_sha=sha, external=tmp / "ext.ipynb", src_dir=src, build=tmp / "build.py")

    def run_audit(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.build(tmp, **kw)
            result = audit_fork.audit("arc-2026-fork-perfpatch-v1", **args)
            written = json.loads((args["e_dir"] / "audit-arc-2026-fork-perfpatch-v1.json").read_text())
            self.assertEqual(result["gates"], written["gates"])
            self.assertEqual(written["kernel"], "f0909172434/arc-2026-fork-perfpatch-v1")
            self.assertIsNone(written["official_score"])
            return written

    def test_all_pass(self):
        r = self.run_audit()
        self.assertTrue(r["submission_allowed"], r["gates"])
        self.assertEqual(r["facts"]["eval_reload_score"], 1.5)

    def test_plain_text_log(self):
        self.assertTrue(self.run_audit(stream=False)["submission_allowed"])

    def test_submission_failures(self):
        for kind in ("missing", "badgrid", "short", "extra", "none"):
            r = self.run_audit(submission=kind)
            self.assertFalse(r["gates"]["submission_full_coverage_valid"], kind)
            self.assertFalse(r["submission_allowed"], kind)

    def test_log_failures(self):
        r = self.run_audit(log=GOOD_LOG + "Traceback (most recent call last)\n")
        self.assertFalse(r["gates"]["no_task_failures"])
        r = self.run_audit(log=GOOD_LOG.replace("[Rank 3] start!", ""))
        self.assertFalse(r["gates"]["four_workers_started"])
        r = self.run_audit(log=GOOD_LOG.replace("*** Reload score: 1.5", ""))
        self.assertFalse(r["gates"]["final_assembly_completed"])
        r = self.run_audit(log=GOOD_LOG.replace("finished aa4ec2a5", "x"))
        self.assertFalse(r["gates"]["smoke_tasks_finished"])

    def failed(self, **kw):
        r = self.run_audit(**kw)
        self.assertFalse(r["submission_allowed"])
        return {k for k, v in r["gates"].items() if not v}

    def test_test_and_review_gates(self):
        self.assertFalse(self.run_audit(passed=False)["submission_allowed"])
        self.assertFalse(self.run_audit(verdict="NO-GO")["submission_allowed"])
        self.assertTrue(self.run_audit(verdict="GO-with-fixes-applied")["submission_allowed"])

    def test_stale_evidence_fails(self):
        self.assertEqual(self.failed(hs_extra={"src_sha256": {"a.py": "0" * 64}}), {"hidden_shape_test_passed"})
        self.assertEqual(self.failed(hs_extra={"src_sha256": {}}), {"hidden_shape_test_passed"})
        self.assertEqual(self.failed(hs_extra={"notebook_sha256": "0" * 64}), {"hidden_shape_test_passed"})

    def test_non_identical_notebook_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.build(tmp)
            (args["upload"] / "n.ipynb").write_bytes(b"edited")
            r = audit_fork.audit("arc-2026-fork-perfpatch-v1", **args)
            self.assertFalse(r["gates"]["byte_identical_fork"])
            self.assertFalse(r["submission_allowed"])
        # external-src copy differs from the pinned constant (preparation.json not trusted)
        self.assertIn("byte_identical_fork", self.failed(ext_bytes=b"other"))

    def test_output_gates(self):
        self.assertEqual(self.failed(smoke_blank="0934a4d8"), {"smoke_ids_have_real_attempt"})
        self.assertEqual(self.failed(log=GOOD_LOG.replace("1.5", "0.0")), {"reload_score_positive"})

    def test_decoded_files_gate_only_if_present(self):
        names = [f"{t}_0.permute0123456789" for t in audit_fork.SMOKE_IDS]
        r = self.run_audit(decoded=names)
        self.assertTrue(r["submission_allowed"])
        self.assertEqual(r["facts"]["decoded_files_in_download"], 4)
        self.assertEqual(self.failed(decoded=names[:3]), {"decoded_files_cover_smoke_ids"})
        self.assertNotIn("decoded_files_cover_smoke_ids", self.run_audit()["gates"])

    def test_metadata_gates(self):
        for key, value, gate in (("enable_internet", True, "metadata_private_offline_l4"), ("is_private", False, "metadata_private_offline_l4"),
                                 ("machine_shape", "T4", "metadata_private_offline_l4"),
                                 ("competition_sources", [], "metadata_sources_match_build"),
                                 ("model_sources", ["x"], "metadata_sources_match_build"),
                                 ("kernel_sources", ["x"], "metadata_sources_match_build"),
                                 ("docker_image", "x", "metadata_sources_match_build")):
            self.assertEqual(self.failed(meta={**META, key: value}), {gate}, key)

    def test_all_log_files_concatenated(self):
        r = self.run_audit(log=GOOD_LOG.replace("*** Reload score: 1.5", ""), second_log="*** Reload score: 2.0\n")
        self.assertTrue(r["submission_allowed"])
        self.assertEqual(r["facts"]["eval_reload_score"], 2.0)
        self.assertIsNone(r["facts"]["model_licence"])


if __name__ == "__main__":
    unittest.main()
