from __future__ import annotations

import torch
import pytest

from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.interfaces import ChunkTransition

C = 10
ACTION_DIM = 14
STATE_DIM = 2062


def _make_transition() -> ChunkTransition:
    return ChunkTransition(
        state_vec=torch.randn(STATE_DIM),
        exec_chunk=torch.randn(C, ACTION_DIM),
        ref_chunk=torch.randn(C, ACTION_DIM),
        reward_seq=torch.randn(C),
        next_state_vec=torch.randn(STATE_DIM),
        next_ref_chunk=torch.randn(C, ACTION_DIM),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(C),
    )


def test_add_and_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.add(_make_transition())
    assert len(buf) == 1


def test_capacity_limit():
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.add(_make_transition())
    assert len(buf) == 5


def test_total_added_keeps_growing_past_capacity():
    """len() stops growing once the deque is full and starts evicting; callers
    that need "how many transitions were added since X" (e.g. online training's
    UTD scaling) must use total_added instead, or the count silently drops to
    ~0 forever once the buffer fills."""
    buf = ReplayBuffer(capacity=5)
    for _ in range(10):
        buf.add(_make_transition())
    assert len(buf) == 5
    assert buf.total_added == 10


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(_make_transition())

    batch = buf.sample(8)
    assert batch["state_vec"].shape == (8, STATE_DIM)
    assert batch["exec_chunk_flat"].shape == (8, C * ACTION_DIM)
    assert batch["ref_chunk_flat"].shape == (8, C * ACTION_DIM)
    assert batch["reward_seq"].shape == (8, C)
    assert batch["next_state_vec"].shape == (8, STATE_DIM)
    assert batch["next_ref_flat"].shape == (8, C * ACTION_DIM)
    assert batch["done"].shape == (8,)
    assert batch["source"].shape == (8,)
    assert batch["episode_id"].shape == (8,)
    assert batch["is_critical"].shape == (8,)


def test_sample_capped_by_buffer_size():
    buf = ReplayBuffer(capacity=100)
    for _ in range(3):
        buf.add(_make_transition())
    batch = buf.sample(10)
    assert batch["state_vec"].shape[0] == 3


def test_batch_keys():
    buf = ReplayBuffer(capacity=100)
    buf.add(_make_transition())
    batch = buf.sample(1)
    expected_keys = {
        "state_vec", "exec_chunk_flat", "ref_chunk_flat",
        "reward_seq", "next_state_vec", "next_ref_flat", "done", "actual_steps",
        "source", "episode_id", "is_critical", "outcome",
        "intervention_mask_flat",
    }
    assert set(batch.keys()) == expected_keys


def _make_episode_transition(
    episode_id: int, done: bool = False, success: bool = False, intervention: bool = False,
) -> ChunkTransition:
    reward_seq = torch.zeros(C)
    if done and success:
        reward_seq[-1] = 1.0
    return ChunkTransition(
        state_vec=torch.randn(STATE_DIM),
        exec_chunk=torch.randn(C, ACTION_DIM),
        ref_chunk=torch.randn(C, ACTION_DIM),
        reward_seq=reward_seq,
        next_state_vec=torch.randn(STATE_DIM),
        next_ref_chunk=torch.randn(C, ACTION_DIM),
        done=torch.tensor(float(done)),
        intervention=torch.tensor(float(intervention)),
        actual_steps=torch.tensor(C),
        episode_id=torch.tensor(episode_id),
    )


class TestEpisodeOutcomes:
    def test_count_outcomes(self):
        buf = ReplayBuffer(capacity=100)
        # Episode 0: two lead-up transitions + a successful terminal one.
        buf.add(_make_episode_transition(0))
        buf.add(_make_episode_transition(0))
        buf.add(_make_episode_transition(0, done=True, success=True))
        # Episode 1: a single failed terminal transition.
        buf.add(_make_episode_transition(1, done=True, success=False))
        successes, failures = buf.count_outcomes()
        assert successes == 1
        assert failures == 1

    def test_episode_outcomes_tags_non_terminal_transitions_via_episode_id(self):
        buf = ReplayBuffer(capacity=100)
        buf.add(_make_episode_transition(5))
        buf.add(_make_episode_transition(5, done=True, success=True))
        outcomes = buf.episode_outcomes()
        assert outcomes == {5: "success"}

    def test_explicit_failure_overrides_positive_shaping_reward(self):
        buf = ReplayBuffer(capacity=100)
        transition = _make_episode_transition(8, done=True, success=True)
        transition.outcome = torch.tensor(0.0)
        buf.add(transition)
        assert buf.episode_outcomes() == {8: "failure"}


class TestOutcomeLabels:
    def test_maps_success_failure_and_unresolved(self):
        buf = ReplayBuffer(capacity=100)
        buf.add(_make_episode_transition(1, done=True, success=True))
        buf.add(_make_episode_transition(2, done=True, success=False))
        buf.add(_make_episode_transition(3))  # no terminal transition yet
        episode_id = torch.tensor([1, 2, 3, 999])  # 999 unknown to the buffer
        labels = buf.outcome_labels(episode_id)
        assert labels.tolist() == [1.0, 0.0, -1.0, -1.0]


class TestStratifiedSampling:
    @staticmethod
    def _tiny_pool_buffer() -> ReplayBuffer:
        """500 plain transitions plus one success, one failure, five interventions.

        The three interesting buckets are far smaller than a 40/30/20% quota on a
        100-batch, which is exactly the regime allow_resample governs.
        """
        buf = ReplayBuffer(capacity=1000)
        for eid in range(50, 550):
            buf.add(_make_episode_transition(eid))
        buf.add(_make_episode_transition(1, done=True, success=True))
        buf.add(_make_episode_transition(2, done=True, success=False))
        for _ in range(5):
            buf.add(_make_episode_transition(3, intervention=True))
        return buf

    @staticmethod
    def _count_successes(batch) -> float:
        return (batch["done"] * (batch["reward_seq"].sum(dim=-1) > 0).float()).sum().item()

    def test_backfills_from_other_bucket_when_buffer_is_small(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(3):
            buf.add(_make_episode_transition(0))
        # Only 3 transitions exist, so without resampling the batch comes up short.
        # (It can still exceed 3: a transition counts toward both the 'recent' and
        # 'other' buckets, and cross-bucket overlap was always allowed.)
        assert buf.sample_stratified(8)["state_vec"].shape[0] < 8
        # With resampling the quota is filled by repeating them.
        assert buf.sample_stratified(8, allow_resample=True)["state_vec"].shape[0] == 8

    def test_allow_resample_fills_a_tiny_quota_by_repeating(self):
        buf = self._tiny_pool_buffer()
        batch = buf.sample_stratified(
            100, success_frac=0.4, failure_frac=0.3, intervention_frac=0.2,
            recent_frac=0.1, allow_resample=True,
        )
        assert batch["state_vec"].shape[0] == 100
        # The single success transition is repeated up to its 40% quota.
        assert self._count_successes(batch) >= 30

    def test_default_draws_each_bucket_at_most_once(self):
        buf = self._tiny_pool_buffer()
        batch = buf.sample_stratified(
            100, success_frac=0.4, failure_frac=0.3, intervention_frac=0.2, recent_frac=0.1,
        )
        # Still a full batch -- the shortfall path tops it up uniformly.
        assert batch["state_vec"].shape[0] == 100
        # But the one success transition appears once, not 40 times: a tiny pool
        # must not stand in for a fixed share of every batch. See ReplayBuffer's
        # docstring for the amplification this avoids.
        assert self._count_successes(batch) <= 2

    def test_full_buckets_still_meet_their_quota_without_resampling(self):
        buf = ReplayBuffer(capacity=2000)
        for eid in range(100, 400):
            buf.add(_make_episode_transition(eid))
        for eid in range(400, 500):
            buf.add(_make_episode_transition(eid, done=True, success=True))
        batch = buf.sample_stratified(
            100, success_frac=0.4, failure_frac=0.0, intervention_frac=0.0, recent_frac=0.0,
        )
        assert batch["state_vec"].shape[0] == 100
        # 100 success transitions comfortably cover a 40-slot quota.
        assert self._count_successes(batch) >= 40

    def test_batch_keys_match_uniform_sample(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.add(_make_episode_transition(0))
        assert set(buf.sample_stratified(8).keys()) == set(buf.sample(8).keys())
