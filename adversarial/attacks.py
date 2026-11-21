"""Mimicry attacks on the account stats of bots and trolls.

Every attack moves one or more features of each inauthentic account toward the value of a randomly
chosen real user ("donor") from the same population's training split:

    t(x') = (1 - s) * t(x) + s * t(donor)

where t is a transform that keeps the feature in its valid range (log for counts and ages, logit for
fractions, atanh for burstiness) and s is the attack strength (0 = none, 0.5 = mild, 1 = strong).
At s = 1 the attacked feature of the fakes has exactly the distribution of real users' values, so it
carries no information at all. Real users are never changed.

Each inauthentic account uses ONE donor for every feature an attack touches, so the copied values
stay realistic together (a fake copies one real person's habits, not a mix of several people's).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRENGTHS = {"none": 0.0, "mild": 0.5, "strong": 1.0}

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# feature -> (forward transform, inverse transform)
TRANSFORMS = {
    "posts_per_day": (np.log, np.exp),
    "account_age_days": (np.log, np.exp),
    "reply_fraction": (_logit, _sigmoid),
    "url_fraction": (_logit, _sigmoid),
    "night_activity_frac": (_logit, _sigmoid),
    "burstiness": (lambda x: np.arctanh(np.clip(x, -1 + EPS, 1 - EPS)), np.tanh),
}


@dataclass(frozen=True)
class Attack:
    key: str
    name: str
    features: tuple  # features the attacker changes (posts_per_week follows posts_per_day)
    cost: str  # "cheap", "moderate" or "expensive" for a real operator
    how: str  # what the operator would actually do


SINGLE_ATTACKS = (
    Attack("posting_rate", "Post less often", ("posts_per_day",), "cheap",
           "Throttle the bots' posting rate to a normal user's; costs reach, not money."),
    Attack("replies", "Reply more", ("reply_fraction",), "cheap",
           "Have the bots reply to others as often as people do; automated replies are easy."),
    Attack("links", "Share fewer links", ("url_fraction",), "cheap",
           "Post fewer links per post; costs click-throughs, not money."),
    Attack("night", "Post in the daytime", ("night_activity_frac",), "cheap",
           "Schedule posts for the hours people are awake; one setting in a scheduler."),
    Attack("burstiness", "Post steadily", ("burstiness",), "moderate",
           "Spread posts out instead of firing them in bursts; needs smarter scheduling."),
    Attack("account_age", "Use aged accounts", ("account_age_days",), "expensive",
           "Buy old accounts or wait months before using new ones; costs money or time."),
)
COMBINED = Attack(
    "combined", "All cheap and moderate changes",
    tuple(f for a in SINGLE_ATTACKS if a.cost != "expensive" for f in a.features), "moderate",
    "Every change above except buying aged accounts, which a typical operator avoids.",
)
ATTACKS = {a.key: a for a in (*SINGLE_ATTACKS, COMBINED)}


def pick_donors(accounts: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """For every account, the row index of a random real user in the training split (used only for fakes)."""
    pool = np.flatnonzero((accounts["label"].to_numpy() == 0) & (accounts["split"].to_numpy() == "train"))
    return pool[rng.integers(0, len(pool), size=len(accounts))]


def apply_attack(accounts: pd.DataFrame, attack: Attack, strength: float, donors: np.ndarray) -> pd.DataFrame:
    """Return a copy of `accounts` with the attack applied to every inauthentic account."""
    out = accounts.copy()
    if strength == 0:
        return out
    fake = out["label"].to_numpy() == 1
    for feat in attack.features:
        fwd, inv = TRANSFORMS[feat]
        x = out[feat].to_numpy(float)
        d = x[donors]
        new = inv((1 - strength) * fwd(x[fake]) + strength * fwd(d[fake]))
        if feat == "posts_per_day":
            # keep the weekly count consistent with the new daily rate (same per-account noise ratio)
            ratio = out["posts_per_week"].to_numpy(float)[fake] / x[fake]
            out.loc[fake, "posts_per_week"] = new * ratio
        if feat == "account_age_days":
            new = np.clip(new, 1, 4000)
        out.loc[fake, feat] = new
    return out


def changed_features(attack: Attack) -> tuple:
    """Columns an attack modifies (posting rate also moves the weekly count)."""
    extra = ("posts_per_week",) if "posts_per_day" in attack.features else ()
    return tuple(attack.features) + extra
