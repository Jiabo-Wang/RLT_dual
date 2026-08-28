"""Low-memory replacement for ``PI05Policy.from_pretrained``.

Upstream ``modeling_pi05.py`` loads a checkpoint like this::

    model = cls(config)                       # fp32 random init of 4.14B params
    original_state_dict = load_file(...)      # then immediately overwritten
    model.load_state_dict(remapped, strict=strict)

The random init is pure waste -- every value it produces is overwritten by the
checkpoint on the next line -- but it costs a **23.7 GiB transient and 116 s** for
a pi0.5 SFT checkpoint. On the 30 GiB deployment box that is enough to trip the
global OOM killer, which takes the editor down with it when the run was launched
from an IDE terminal (the process sits in the IDE's cgroup).

This module swaps in the standard fix: build the skeleton with parameters on the
``meta`` device (no allocation), then hand the checkpoint's own tensors straight
to ``load_state_dict(..., assign=True)``. Same weights, same key remapping, same
strictness -- measured at **9.7 GiB peak and 1.1 s** instead.

Buffers are deliberately kept real (``include_buffers=False``): a handful of them
(``rotary_emb.inv_freq``, ``position_ids``) are computed in ``__init__`` rather
than stored in the checkpoint, so putting them on ``meta`` would leave them
dangling with nothing to assign.

Call :func:`install` once before ``make_policy``. It is idempotent and only
touches ``pi05``; every other policy type keeps its normal loading path.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)

_installed = False


def _load_low_mem(cls, pretrained_name_or_path, *, config=None, strict: bool = True, **kwargs: Any):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    from lerobot.configs.policies import PreTrainedConfig

    if pretrained_name_or_path is None:
        raise ValueError("pretrained_name_or_path is required")

    if config is None:
        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path, **kwargs
        )

    # Resolve the weights first: if this fails there is no point building anything.
    resolved_file = cached_file(
        pretrained_name_or_path,
        "model.safetensors",
        cache_dir=kwargs.get("cache_dir"),
        force_download=kwargs.get("force_download", False),
        resume_download=kwargs.get("resume_download"),
        proxies=kwargs.get("proxies"),
        token=kwargs.get("token"),
        revision=kwargs.get("revision"),
        local_files_only=kwargs.get("local_files_only", False),
    )

    # PI05Policy.__init__ ends with `self.model.to(config.device)`, which raises on
    # meta parameters. Neutralise `.to` for the duration of the constructor -- scoped
    # to PI05Pytorch, not nn.Module, so nothing else in the process is affected.
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    had_own_to = "to" in PI05Pytorch.__dict__
    prev_to = PI05Pytorch.__dict__.get("to")
    PI05Pytorch.to = lambda self, *a, **k: self
    try:
        with init_empty_weights(include_buffers=False):
            model = cls(config)
    finally:
        if had_own_to:
            PI05Pytorch.to = prev_to
        else:
            del PI05Pytorch.to

    state_dict = load_file(resolved_file)

    # Same two remap stages as upstream: openpi key fixes, then a "model." prefix.
    fixed_state_dict = model._fix_pytorch_state_dict_keys(state_dict, model.config)
    remapped_state_dict = {}
    remap_count = 0
    for key, value in fixed_state_dict.items():
        if not key.startswith("model."):
            remapped_state_dict[f"model.{key}"] = value
            remap_count += 1
        else:
            remapped_state_dict[key] = value
    if remap_count > 0:
        log.info("Remapped %d state dict keys", remap_count)

    missing_keys, unexpected_keys = model.load_state_dict(
        remapped_state_dict, strict=strict, assign=True
    )
    del state_dict, fixed_state_dict, remapped_state_dict

    if missing_keys:
        log.warning("Missing keys when loading pi05 state dict: %s", missing_keys[:5])
    if unexpected_keys:
        log.warning("Unexpected keys when loading pi05 state dict: %s", unexpected_keys[:5])

    # Anything still on meta got no tensor assigned and would fail at the first
    # forward with a confusing error -- surface it here instead.
    stranded = [
        name
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_meta
    ]
    if stranded:
        raise RuntimeError(
            f"{len(stranded)} pi05 tensors are still on the meta device after loading, "
            f"e.g. {stranded[:5]}. The checkpoint does not cover them; fall back to the "
            "stock loader by not calling evo_rlt.adapters.lerobot.pi05_low_mem_load.install()."
        )

    model.to(config.device)
    if not missing_keys and not unexpected_keys:
        log.info("pi05 checkpoint loaded (low-memory path), all keys matched")
    return model


def install() -> bool:
    """Patch ``PI05Policy.from_pretrained`` onto the low-memory path. Idempotent.

    Returns True if the patch is active, False if pi05 or accelerate is unavailable
    (in which case the stock loader stays in place).
    """
    global _installed
    if _installed:
        return True
    try:
        from accelerate import init_empty_weights  # noqa: F401

        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        log.debug("pi05 low-memory loader not installed: %s", exc)
        return False

    PI05Policy._from_pretrained_stock = PI05Policy.from_pretrained
    PI05Policy.from_pretrained = classmethod(_load_low_mem)
    _installed = True
    log.info("pi05 low-memory checkpoint loader installed")
    return True
