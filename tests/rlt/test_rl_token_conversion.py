"""Converting an evo-rlt-train-rl-token checkpoint into an RLTokenPolicy directory.

Two RL-token artefact formats coexist and only the directory one feeds online RL.
Handing ChunkACPolicy the .pt file makes huggingface_hub read the path as a repo
id and report "Repo id must be in the form 'repo_name' or 'namespace/repo_name'",
which says nothing about the format being wrong.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diagnostics.convert_rl_token_checkpoint import convert, verify  # noqa: E402

TOKEN_DIM = 8  # tiny, so the fixture stays fast


def _fake_checkpoint(tmp_path, *, decoder=True, num_rl_tokens=1):
    state = {"rl_token_embed": torch.randn(num_rl_tokens, TOKEN_DIM)}
    state["encoder.layers.0.linear1.weight"] = torch.randn(4, TOKEN_DIM)
    if decoder:
        state["decoder.layers.0.linear1.weight"] = torch.randn(4, TOKEN_DIM)
    src = tmp_path / "demo_adapt_checkpoint.pt"
    torch.save(
        {"rl_token_state_dict": state, "step": 10000, "losses": [1.0, 0.5],
         "metadata": {"num_rl_tokens": num_rl_tokens, "norm_gamma": 0.5,
                      "norm_stats": "/on/another/machine/crp_token_std.pt"}},
        src,
    )
    return src, state


def _convert(tmp_path, src, **kw):
    out = tmp_path / "policy_dir"
    convert(src, out, vla_path=str(tmp_path / "vla"), tokenizer_path="google/paligemma-3b-pt-224",
            camera_keys=["top", "left_wrist", "right_wrist"],
            action_dim=14, proprio_dim=14, chunk_size=50, **kw)
    return out


class TestConversion:
    def test_writes_the_two_files_lerobot_looks_for(self, tmp_path):
        src, _ = _fake_checkpoint(tmp_path)
        out = _convert(tmp_path, src)
        assert (out / "config.json").is_file()
        assert (out / "model.safetensors").is_file()

    def test_every_tensor_survives_under_the_rl_token_prefix(self, tmp_path):
        from safetensors.torch import load_file

        src, state = _fake_checkpoint(tmp_path)
        written = load_file(_convert(tmp_path, src) / "model.safetensors")
        assert set(written) == {f"rl_token.{k}" for k in state}
        for key, value in state.items():
            assert torch.equal(written[f"rl_token.{key}"], value)

    def test_config_round_trips_through_the_loader_chunkacpolicy_uses(self, tmp_path):
        src, _ = _fake_checkpoint(tmp_path)
        out = _convert(tmp_path, src)
        verify(out, src)  # raises on any mismatch

    def test_architecture_is_read_off_the_weights_not_assumed(self, tmp_path):
        from lerobot.configs.policies import PreTrainedConfig

        src, _ = _fake_checkpoint(tmp_path, num_rl_tokens=3)
        cfg = PreTrainedConfig.from_pretrained(_convert(tmp_path, src))
        # ChunkACPolicy validates exactly these two against its own config.
        assert cfg.rl_token_dim == TOKEN_DIM
        assert cfg.rl_token_num_rl_tokens == 3

    def test_the_training_machines_norm_stats_path_is_dropped(self, tmp_path):
        from lerobot.configs.policies import PreTrainedConfig

        src, _ = _fake_checkpoint(tmp_path)
        cfg = PreTrainedConfig.from_pretrained(_convert(tmp_path, src))
        # _dim_std is a non-persistent buffer that inference never reads, and the
        # recorded path points at a file on the training box that would fail to load.
        assert cfg.norm_stats_path is None
        assert cfg.norm_gamma == 0.5  # kept, for provenance

    def test_the_source_file_is_left_untouched(self, tmp_path):
        src, _ = _fake_checkpoint(tmp_path)
        before = src.read_bytes()
        _convert(tmp_path, src)
        assert src.read_bytes() == before


class TestRejections:
    def test_an_encoder_only_checkpoint_is_refused(self, tmp_path):
        # RLTokenPolicy builds the module with inference_only=False, so a
        # decoder-less state dict would load short and say nothing.
        src, _ = _fake_checkpoint(tmp_path, decoder=False)
        with pytest.raises(ValueError, match="decoder"):
            _convert(tmp_path, src)

    def test_an_unrelated_torch_file_is_refused(self, tmp_path):
        src = tmp_path / "other.pt"
        torch.save({"model_state_dict": {}}, src)
        with pytest.raises(ValueError, match="rl_token_state_dict"):
            _convert(tmp_path, src)
