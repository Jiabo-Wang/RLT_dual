"""Convert an ``evo-rlt-train-rl-token`` checkpoint into an RLTokenPolicy directory.

    python diagnostics/convert_rl_token_checkpoint.py \\
        crp_rl_token/demo_adapt_checkpoint.pt \\
        outputs/crp_rl_token_policy \\
        --vla-path pretrained_model

Two RL-token artefact formats coexist in this repo and only one of them feeds
online RL:

  evo-rlt-train-rl-token  ->  demo_adapt_checkpoint.pt
      a plain torch save: {"rl_token_state_dict", "step", "losses", "metadata"}.
      Consumed by build_pi05_policy(rl_token_checkpoint=...), i.e. the standalone
      CLIs (train_chunk_actor_critic, compute_token_variance).

  lerobot-train --policy.type=rlt_token  ->  a checkpoint *directory*
      config.json + model.safetensors, loaded by ChunkACPolicy through
      PreTrainedConfig.from_pretrained(). Handing it the .pt file instead makes
      huggingface_hub read the path as a repo id and report
      "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".

The weights themselves are interchangeable: RLTokenPolicy holds exactly the
``RLTokenModule(..., inference_only=False)`` that the standalone trainer trains,
and it keeps pi0.5 outside module tracking (``object.__setattr__``), so its
state_dict is the same tensors under an ``rl_token.`` prefix. This rewrites the
container, so a finished run does not have to be repeated under the other tool.

Nothing is read from the network and the source file is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def convert(
    src: Path,
    out_dir: Path,
    vla_path: str,
    tokenizer_path: str,
    camera_keys: list[str],
    action_dim: int,
    proprio_dim: int,
    chunk_size: int,
) -> None:
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    blob = torch.load(src, map_location="cpu", weights_only=False)
    if "rl_token_state_dict" not in blob:
        raise ValueError(
            f"{src} has keys {sorted(blob)} -- expected a demo_adapt_checkpoint.pt "
            "produced by evo-rlt-train-rl-token, which stores 'rl_token_state_dict'."
        )
    state = blob["rl_token_state_dict"]
    meta = blob.get("metadata") or {}
    print(f"源: {src}")
    print(f"  步数 {blob.get('step')} | {len(state)} 个张量 | "
          f"{sum(v.numel() for v in state.values()) / 1e6:.1f}M 参数")
    if meta:
        print(f"  metadata: {json.dumps(meta, ensure_ascii=False)}")

    # The decoder is only used by the reconstruction loss, but RLTokenPolicy builds
    # the module with inference_only=False, so its state_dict expects the decoder
    # keys too. Keeping them means the directory can also resume training.
    has_decoder = any(k.startswith("decoder.") for k in state)
    if not has_decoder:
        raise ValueError(
            "This checkpoint has no decoder weights, so it came from an "
            "inference-only module; RLTokenPolicy expects the full encoder+decoder."
        )

    num_rl_tokens = int(meta.get("num_rl_tokens", 1))
    token_dim = int(state["rl_token_embed"].shape[-1])
    print(f"  推断出 token_dim={token_dim} num_rl_tokens={num_rl_tokens}")

    config = RLTokenPolicyConfig(
        vla_pretrained_path=vla_path,
        rl_token_dim=token_dim,
        rl_token_num_rl_tokens=num_rl_tokens,
        # norm_gamma / norm_stats only shape the reconstruction loss. Recording the
        # gamma keeps the run reproducible; the stats path is deliberately dropped
        # because _dim_std is a non-persistent buffer that inference never reads,
        # and pointing at a file from the training machine would just fail to load.
        norm_gamma=float(meta.get("norm_gamma", 0.0)),
        norm_stats_path=None,
        chunk_size=chunk_size,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        camera_keys=list(camera_keys),
        tokenizer_path=tokenizer_path,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out_dir)
    save_file(
        {f"rl_token.{k}": v.contiguous() for k, v in state.items()},
        out_dir / "model.safetensors",
    )

    print(f"\n写入 {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1024**2:8.1f} MB")


def verify(out_dir: Path, src: Path) -> None:
    """Reload through the same call ChunkACPolicy makes, and diff the tensors."""
    from safetensors.torch import load_file

    from lerobot.configs.policies import PreTrainedConfig

    from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    cfg = PreTrainedConfig.from_pretrained(out_dir)
    if not isinstance(cfg, RLTokenPolicyConfig):
        raise TypeError(f"config.json round-tripped as {type(cfg).__name__}")
    print("\n校验")
    print(f"  PreTrainedConfig.from_pretrained -> {type(cfg).__name__}"
          f" (type={cfg.type}, dim={cfg.rl_token_dim}, tokens={cfg.rl_token_num_rl_tokens})")

    original = torch.load(src, map_location="cpu", weights_only=False)["rl_token_state_dict"]
    written = load_file(out_dir / "model.safetensors")
    assert len(written) == len(original), f"{len(written)} vs {len(original)} 个张量"
    for key, value in original.items():
        assert torch.equal(written[f"rl_token.{key}"], value), f"{key} 不一致"
    print(f"  {len(written)} 个张量与源文件逐比特一致")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--vla-path", required=True,
                    help="pi0.5 SFT checkpoint dir (ChunkACPolicy overrides this at "
                         "load time, but the value is recorded for provenance)")
    ap.add_argument("--tokenizer-path", default="google/paligemma-3b-pt-224")
    ap.add_argument("--cameras", default="top,left_wrist,right_wrist")
    ap.add_argument("--action-dim", type=int, default=14)
    ap.add_argument("--proprio-dim", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=50)
    args = ap.parse_args()

    convert(
        args.src.expanduser().resolve(),
        args.out_dir.expanduser().resolve(),
        str(Path(args.vla_path).expanduser().resolve()),
        args.tokenizer_path,
        [c.strip() for c in args.cameras.split(",") if c.strip()],
        args.action_dim,
        args.proprio_dim,
        args.chunk_size,
    )
    verify(args.out_dir.expanduser().resolve(), args.src.expanduser().resolve())
    print("\n用它替换 --rl-token-path 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
