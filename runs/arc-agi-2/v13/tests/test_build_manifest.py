"""Unit tests for model profile sha_manifest bundling in build.py (no network, no competition data)."""
import copy
import json
import sys
from pathlib import Path

import nbformat as nbf
import pytest

HERE = Path(__file__).resolve().parent
V13_DIR = HERE.parent
if str(V13_DIR) not in sys.path:
    sys.path.insert(0, str(V13_DIR))
if str(V13_DIR / "src7") not in sys.path:
    sys.path.insert(0, str(V13_DIR / "src7"))

import build as B  # noqa: E402
import model_profiles as MP  # noqa: E402


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    root = tmp_path
    v11_src = root / "runs/arc-agi-2/v11/src"
    v13_dir = root / "runs/arc-agi-2/v13"
    src_dir = v13_dir / "src"
    src7_dir = v13_dir / "src7"
    v11_src.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    src7_dir.mkdir(parents=True)

    for name in B.NVARC_FILES:
        data = f"# nvarc {name}\nVALUE = {len(name)}\n".encode()
        (v11_src / name).write_bytes(data)
        (src_dir / name).write_bytes(data)

    for name in B.V13_FILES:
        if name.endswith(".py"):
            data = f"# v13 {name}\nVALUE = {len(name)}\n".encode()
        else:
            data = (json.dumps({"name": name}) + "\n").encode()
        target_dir = src7_dir if name in ("model_profiles.py", "qwen38_fp8_sha256.json") else src_dir
        (target_dir / name).write_bytes(data)

    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(B, "HERE", v13_dir)
    monkeypatch.setattr(B, "SRC", src_dir)
    monkeypatch.setattr(B, "V11_SRC", v11_src)
    monkeypatch.setattr(MP, "PROFILES", copy.deepcopy(MP.PROFILES))
    return v13_dir


def test_default_profile_source_sha256_unchanged(fake_workspace):
    src_dir = fake_workspace / "src"
    src7_dir = fake_workspace / "src7"
    legacy_sources = {
        name: ((src_dir / name) if (src_dir / name).exists() else (src7_dir / name)).read_bytes()
        for name in B.FILES
    }
    expected_source_sha256 = {name: B.sha(data) for name, data in legacy_sources.items()}

    out = B.build("arc-2026-v13-default-check", out=fake_workspace / "upload-default")
    prep = json.loads((out / "preparation.json").read_text())
    meta = json.loads((out / "kernel-metadata.json").read_text())
    nb = nbf.read(out / "arc-2026-v13-default-check.ipynb", as_version=4)

    assert B.profile_files(MP.DEFAULT_PROFILE) == B.FILES
    assert prep["source_sha256"] == expected_source_sha256
    assert tuple(prep["source_sha256"]) == B.FILES
    writefiles = [c.source.split('\n', 1)[0][len('%%writefile '):] for c in nb.cells if c.source.startswith('%%writefile ')]
    assert tuple(writefiles) == B.FILES
    assert len(nb.cells) == 22  # default-profile notebook: 9 fixed cells + 13 files (pre-change build)
    assert meta["model_sources"] == [B.NVARC_MODEL, MP.PROFILES[MP.DEFAULT_PROFILE]["model_source"]]


def test_launchable_profile_packs_sha_manifest(fake_workspace):
    src_dir = fake_workspace / "src"
    src7_dir = fake_workspace / "src7"

    fake_prof = copy.deepcopy(MP.PROFILES["qwen4_27b_template"])
    fake_prof.update({
        "name": "qwen4_27b_fp8_fake",
        "launchable": True,
        "model_source": "fake/qwen4-27b-fp8/PyTorch/hf-fp8/1",
        "sha_manifest": "qwen4_fp8_sha256.json",
        "vllm_version": "0.19.0",
        "reasoning_parser": "qwen3",
        "quantization": "fp8",
        "think_tags": ("<think>", "</think>"),
    })
    MP.PROFILES["qwen4_27b_fp8_fake"] = fake_prof

    src7_manifest_bytes = b'{"layers-0.safetensors": "1111"}\n'
    (src7_dir / "qwen4_fp8_sha256.json").write_bytes(src7_manifest_bytes)

    out_fallback = B.build(
        "arc-2026-v13-fake-src7",
        out=fake_workspace / "upload-fake-src7",
        model_profile="qwen4_27b_fp8_fake",
    )
    prep_fallback = json.loads((out_fallback / "preparation.json").read_text())
    nb_fallback = nbf.read(out_fallback / "arc-2026-v13-fake-src7.ipynb", as_version=4)

    assert "qwen4_fp8_sha256.json" in prep_fallback["source_sha256"]
    assert prep_fallback["source_sha256"]["qwen4_fp8_sha256.json"] == B.sha(src7_manifest_bytes)
    assert tuple(prep_fallback["source_sha256"]) == B.FILES + ("qwen4_fp8_sha256.json",)
    assert any(
        c.cell_type == "code" and c.source.startswith("%%writefile qwen4_fp8_sha256.json\n")
        for c in nb_fallback.cells
    )

    src_manifest_bytes = b'{"layers-0.safetensors": "2222"}\n'
    (src_dir / "qwen4_fp8_sha256.json").write_bytes(src_manifest_bytes)

    out_primary = B.build(
        "arc-2026-v13-fake-src",
        out=fake_workspace / "upload-fake-src",
        model_profile="qwen4_27b_fp8_fake",
    )
    prep_primary = json.loads((out_primary / "preparation.json").read_text())
    assert prep_primary["source_sha256"]["qwen4_fp8_sha256.json"] == B.sha(src_manifest_bytes)


def test_template_and_todo_manifest_profiles_preserve_behavior(fake_workspace):
    out_tpl = B.build(
        "arc-2026-v13-template-check",
        out=fake_workspace / "upload-template",
        model_profile="qwen4_27b_template",
    )
    prep_tpl = json.loads((out_tpl / "preparation.json").read_text())
    assert B.profile_files("qwen4_27b_template") == B.FILES
    assert tuple(prep_tpl["source_sha256"]) == B.FILES
    assert MP.TODO_FILL not in prep_tpl["source_sha256"]

    todo_manifest_prof = copy.deepcopy(MP.PROFILES[MP.DEFAULT_PROFILE])
    todo_manifest_prof["name"] = "qwen4_todo_manifest"
    todo_manifest_prof["sha_manifest"] = f"{MP.TODO_FILL}_sha256.json"
    MP.PROFILES["qwen4_todo_manifest"] = todo_manifest_prof

    out_todo = B.build(
        "arc-2026-v13-todo-manifest-check",
        out=fake_workspace / "upload-todo-manifest",
        model_profile="qwen4_todo_manifest",
    )
    prep_todo = json.loads((out_todo / "preparation.json").read_text())
    assert B.profile_files("qwen4_todo_manifest") == B.FILES
    assert tuple(prep_todo["source_sha256"]) == B.FILES
    assert todo_manifest_prof["sha_manifest"] not in prep_todo["source_sha256"]


def test_gate_accepts_fresh_default_and_rejects_non_launchable_profile(tmp_path):
    """Real src7 build (no fake workspace): gate() must pass for the default profile and refuse a TODO_FILL template profile."""
    L = pytest.importorskip("launch_v13", reason="Kaggle launch/quota layer (launch_v13.py) is not part of the public release")
    out = B.build("arc-2026-v13-gate-check", src="src7", out=tmp_path / "upload-gate-default")
    L.gate(out)
    tpl = B.build("arc-2026-v13-gate-tpl", src="src7", out=tmp_path / "upload-gate-tpl", model_profile="qwen4_27b_template")
    with pytest.raises(ValueError, match="model profile not launchable"):
        L.gate(tpl)
