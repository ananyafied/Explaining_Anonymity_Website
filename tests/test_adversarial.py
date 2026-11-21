"""Tests for the adversarial dataset generator (adversarial/). Builds populations in memory."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from adversarial.attacks import ATTACKS, STRENGTHS, changed_features
from adversarial.generate import build
from anonsynth.config import SynthConfig
from anonsynth.generate import generate
from anonsynth.tabular import FEATURES

ATTACKED_FEATURES = sorted({c for a in ATTACKS.values() for c in changed_features(a)})


@pytest.fixture(scope="module")
def adv():
    return build(seed=1, n_accounts=5000)


@pytest.fixture(scope="module")
def original():
    return generate(SynthConfig(seed=0))


def test_same_format_as_the_original_data(adv, original):
    pop, _, variants = adv
    assert list(pop.accounts.columns) == list(original.accounts.columns)
    assert list(pop.posts.columns) == list(original.posts.columns)
    assert list(pop.edges.columns) == list(original.edges.columns)
    assert pop.seq_counts.shape == original.seq_counts.shape
    assert list(pop.truth.columns) == list(original.truth.columns)
    for name, (acc, _) in variants.items():
        assert list(acc.columns) == list(pop.accounts.columns), name
        assert acc[FEATURES].notna().all().all(), name


def test_variants_cover_every_attack_and_nonzero_strength(adv):
    _, _, variants = adv
    expected = {f"{k}_{s}" for k in ATTACKS for s, v in STRENGTHS.items() if v > 0}
    assert set(variants) == expected


def test_deterministic():
    a = build(seed=3, n_accounts=600)
    b = build(seed=3, n_accounts=600)
    pd.testing.assert_frame_equal(a[0].accounts, b[0].accounts)
    assert (a[1] == b[1]).all()
    for k in a[2]:
        pd.testing.assert_frame_equal(a[2][k][0], b[2][k][0])
    c = build(seed=4, n_accounts=600)
    assert not a[0].accounts[FEATURES].equals(c[0].accounts[FEATURES])


def test_trending_is_gone(adv):
    pop, _, _ = adv
    assert not pop.posts["text"].str.contains("#trending", regex=False).any()
    assert not pop.posts_truth["has_spurious_token"].any()


def test_labels_and_real_users_never_change(adv):
    pop, _, variants = adv
    real = pop.accounts["label"] == 0
    for name, (acc, _) in variants.items():
        assert (acc["label"] == pop.accounts["label"]).all(), name
        assert (acc["split"] == pop.accounts["split"]).all(), name
        pd.testing.assert_frame_equal(acc[real], pop.accounts[real], obj=name)


def test_real_users_look_like_the_original_real_users(adv, original):
    """A fresh population from the same rules: real users' stats should match the original's statistically."""
    pop, _, _ = adv
    a = pop.accounts[pop.accounts["label"] == 0]
    b = original.accounts[original.accounts["label"] == 0]
    for f in FEATURES:
        assert ks_2samp(a[f], b[f]).pvalue > 0.001, f


def test_attacks_move_only_their_target_features(adv):
    pop, _, variants = adv
    fake = pop.accounts["label"] == 1
    for key, attack in ATTACKS.items():
        for s in ("mild", "strong"):
            acc, truth = variants[f"{key}_{s}"]
            targets = set(changed_features(attack))
            for f in FEATURES:
                same = np.allclose(acc.loc[fake, f], pop.accounts.loc[fake, f])
                assert same != (f in targets), f"{key}_{s}: {f}"
            for f in targets:
                assert np.allclose(truth.loc[fake, f"original_{f}"], pop.accounts.loc[fake, f])


def test_attack_strength_moves_fakes_toward_real_users(adv):
    pop, _, variants = adv
    acc0 = pop.accounts
    fake, real = acc0["label"] == 1, acc0["label"] == 0
    for key, attack in ATTACKS.items():
        for f in attack.features:
            real_med, fake_med = acc0.loc[real, f].median(), acc0.loc[fake, f].median()
            mild = variants[f"{key}_mild"][0].loc[fake, f].median()
            strong = variants[f"{key}_strong"][0].loc[fake, f]
            # mild lands strictly between the fakes' original median and the real users' median
            assert min(fake_med, real_med) < mild < max(fake_med, real_med), f"{key}: {f}"
            # strong: the fakes' values follow the real users' distribution
            assert ks_2samp(strong, acc0.loc[real, f]).pvalue > 0.01, f"{key}: {f}"


def test_posting_rate_keeps_weekly_count_consistent(adv):
    pop, _, variants = adv
    fake = pop.accounts["label"] == 1
    for s in ("mild", "strong"):
        acc = variants[f"posting_rate_{s}"][0]
        before = pop.accounts.loc[fake, "posts_per_week"] / pop.accounts.loc[fake, "posts_per_day"]
        after = acc.loc[fake, "posts_per_week"] / acc.loc[fake, "posts_per_day"]
        assert np.allclose(before, after)


def test_no_leaks_between_splits(adv):
    pop, donors, variants = adv
    acc = pop.accounts
    assert acc["account_id"].is_unique
    # linked accounts of one author never straddle two splits
    by_author = pop.truth.assign(split=acc["split"]).groupby("author_id")["split"].nunique()
    assert (by_author == 1).all()
    # attackers only copy real users from the training split, so test-split real users are never copied
    fake = acc["label"].to_numpy() == 1
    used = donors[fake]
    assert (acc["label"].to_numpy()[used] == 0).all()
    assert (acc["split"].to_numpy()[used] == "train").all()
    for name, (_, truth) in variants.items():
        assert (truth.loc[~fake, "donor_id"] == -1).all(), name
