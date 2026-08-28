"""Key paths must not depend on which workspace the operator happened to cd into.

Two workspaces coexist on the training box (~/crp-rlt holds the VLA run,
~/RLT_dual holds this package). Every failure these tests cover was hit for
real: a --config relative path that only resolves from one of them, a stale
--model-path that surfaced as "Repo id must be in the form ...", and a
--norm-stats file written in one workspace and silently not found in the other.
"""

import os
from pathlib import Path

import pytest

from evo_rlt.cli.common import (
    PACKAGED_CONFIG_DIR,
    REPO_ROOT,
    resolve_artifact_path,
    resolve_config_path,
    resolve_model_path,
    warn_on_shadowing_cuda_libs,
)


class TestRepoLayoutConstants:
    def test_constants_point_at_real_directories(self):
        assert (REPO_ROOT / "pyproject.toml").is_file()
        assert PACKAGED_CONFIG_DIR.is_dir()
        assert (PACKAGED_CONFIG_DIR / "crp_dual_rlt.yaml").is_file()


class TestConfigResolution:
    def test_bare_filename_resolves_from_any_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("crp_dual_rlt.yaml") == (
            PACKAGED_CONFIG_DIR / "crp_dual_rlt.yaml"
        ).resolve()

    def test_repo_relative_path_resolves_from_any_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = resolve_config_path("src/evo_rlt/core/configs/crp_dual_rlt.yaml")
        assert resolved.is_file()

    def test_a_file_in_cwd_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local = tmp_path / "crp_dual_rlt.yaml"
        local.write_text("control_hz: 99\n")
        assert resolve_config_path("crp_dual_rlt.yaml") == local.resolve()

    def test_missing_config_lists_what_is_available(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="crp_dual_rlt.yaml"):
            resolve_config_path("no_such_config.yaml")


class TestModelPathResolution:
    def test_missing_local_path_names_the_directory(self, tmp_path):
        stale = tmp_path / "outputs" / "vla_ft" / "checkpoints" / "last" / "pretrained_model"
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_model_path(str(stale))
        message = str(excinfo.value)
        # Must talk about the path, not about Hugging Face repo-id syntax.
        assert "does not exist" in message
        assert str(stale) in message
        assert "Repo id" not in message

    def test_existing_local_path_is_made_absolute(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "pretrained_model"
        ckpt.mkdir()
        monkeypatch.chdir(tmp_path)
        assert resolve_model_path("pretrained_model") == str(ckpt.resolve())

    def test_hub_repo_id_passes_through(self):
        assert resolve_model_path("lerobot/pi05_base") == "lerobot/pi05_base"


class TestArtifactPaths:
    def test_relative_output_is_resolved_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_artifact_path("outputs/x.pt") == (tmp_path / "outputs" / "x.pt").resolve()

    def test_required_input_must_exist(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="norm-stats"):
            resolve_artifact_path("outputs/crp_token_std.pt", must_exist=True, label="norm-stats")


class TestCudaLibraryShadowing:
    def test_clean_environment_reports_nothing(self, monkeypatch):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        assert warn_on_shadowing_cuda_libs() == []

    def test_system_cuda_on_the_path_is_reported(self, tmp_path, monkeypatch):
        lib = tmp_path / "usr-local-cuda" / "lib64"
        lib.mkdir(parents=True)
        (lib / "libcublas.so.12").touch()
        monkeypatch.setenv("LD_LIBRARY_PATH", str(lib))
        assert str(lib) in warn_on_shadowing_cuda_libs()

    def test_torch_own_bundle_is_not_reported(self, monkeypatch):
        torch = pytest.importorskip("torch")
        bundled = Path(torch.__file__).resolve().parent.parent / "nvidia" / "cublas" / "lib"
        if not bundled.is_dir():
            pytest.skip("this torch build does not ship bundled CUDA libraries")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(bundled))
        assert warn_on_shadowing_cuda_libs() == []

    def test_nonexistent_entries_are_ignored(self, monkeypatch):
        monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(["/no/such/dir", ""]))
        assert warn_on_shadowing_cuda_libs() == []
