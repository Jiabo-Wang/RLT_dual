"""Whoever loads pi0.5 must get the low-memory loader, not just the "pi05" type.

The stock PI05Policy.from_pretrained fp32-random-inits 4.14B params before
overwriting them from the checkpoint: a 23.7 GiB transient. backend.py installed
the fix only when the outer policy type was "pi05", so online RL (type "rlt_ac",
which reaches pi0.5 through RLTokenPolicy) still took the stock path and was
OOM-killed at 26.6 GiB on the 30 GiB box. Measured after the fix: 10.6 GiB.
"""

import sys
import types

import pytest


class TestInstallIsReachedFromTheRltPath:
    def test_backbone_loader_installs_before_loading(self, monkeypatch):
        """RLTokenPolicy._load_pi05_backbone is the single point rlt_token and
        rlt_ac pull in pi0.5, so the install has to happen there."""
        from evo_rlt.adapters.lerobot.policies import modeling_rlt_token as m

        order = []

        monkeypatch.setattr(
            "evo_rlt.adapters.lerobot.pi05_low_mem_load.install",
            lambda: order.append("install"),
        )
        monkeypatch.setattr(m, "_load_pi05_config_from_dir",
                            lambda path: types.SimpleNamespace(dtype=None, device=None))

        class FakePI05:
            @staticmethod
            def from_pretrained(*a, **kw):
                order.append("from_pretrained")
                return types.SimpleNamespace(parameters=lambda: iter(()), eval=lambda: None)

        monkeypatch.setattr(m, "PI05Policy", FakePI05)

        config = types.SimpleNamespace(
            vla_pretrained_path="/tmp/x", vla_dtype="bfloat16",
            device="cpu", vla_revision=None, vla_ft_weight=0,
        )
        policy = types.SimpleNamespace(config=config)
        m.RLTokenPolicy._load_pi05_backbone(policy)

        assert order == ["install", "from_pretrained"], (
            "the loader must be installed before pi0.5 is read, not after"
        )


class TestBackendGate:
    @pytest.mark.parametrize("policy_type", ["pi05", "rlt_ac", "rlt_token"])
    def test_every_type_that_loads_pi05_is_covered(self, policy_type):
        import inspect

        from evo_rlt.adapters.lerobot.record import backend

        source = inspect.getsource(backend.record)
        gate = source[source.index("install_pi05_low_mem_loader()") - 400:
                      source.index("install_pi05_low_mem_loader()")]
        assert f'"{policy_type}"' in gate, (
            f"{policy_type} loads pi0.5 but is not in backend.record's install gate"
        )


class TestIdempotence:
    def test_installing_twice_keeps_one_stock_loader(self):
        """_load_pi05_backbone calls install() on every construction, so a second
        call must not wrap the already-patched classmethod again."""
        pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        from evo_rlt.adapters.lerobot.pi05_low_mem_load import install

        assert install() is True
        # classmethod access rebinds each time, so compare the underlying functions.
        first = PI05Policy.from_pretrained.__func__
        stock = PI05Policy._from_pretrained_stock.__func__
        assert install() is True
        assert PI05Policy.from_pretrained.__func__ is first
        # The invariant that matters: a second install must not save the *patched*
        # loader as the stock one, which would make the original unreachable.
        assert PI05Policy._from_pretrained_stock.__func__ is stock
        assert stock is not first
