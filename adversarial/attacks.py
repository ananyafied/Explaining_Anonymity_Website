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


# ======================================================================== hourly activity (Phase 4)
# Each attack works on the 168 hourly post counts of every bot and troll; real users are never changed.
# `masks` (the planted-pattern hours) are updated so explanations can still be scored against them.

ACTIVITY_ATTACKS = {
    "shift_bursts": Attack("shift_bursts", "Move bursts to random times", ("burst timing",), "cheap",
                           "Randomise when the bursts happen, so bots lose their fixed rhythm and trolls stop posting in sync with their ring."),
    "spread_bursts": Attack("spread_bursts", "Spread bursts out", ("burst height",), "cheap",
                            "Spread each burst's extra posts over the neighbouring hours, so no hour is unusually busy."),
    "daily_rhythm": Attack("daily_rhythm", "Fake a daily rhythm", ("background timing",), "cheap",
                           "Make bots' background posting follow a human day (busier in the evening) instead of staying flat."),
    "match_volume": Attack("match_volume", "Post as much as a real user", ("weekly volume",), "cheap",
                           "Thin out (or top up) the week's posts to a real user's weekly total, keeping the shape."),
}
ACTIVITY_COMBINED = Attack("activity_combined", "All activity changes", ("burst timing", "burst height", "background timing", "weekly volume"), "cheap",
                           "Every change above together.")
ACTIVITY_ATTACKS["activity_combined"] = ACTIVITY_COMBINED


def _runs(mask_row):
    """Contiguous runs of True as (start, length)."""
    runs, h, T = [], 0, len(mask_row)
    while h < T:
        if mask_row[h]:
            s = h
            while h < T and mask_row[h]:
                h += 1
            runs.append((s, h - s))
        else:
            h += 1
    return runs


def shift_bursts(c, m, s, rng):
    """Move each planted run (a bot's burst hour, a troll's 3-hour window) to a random free place with probability s."""
    c, m = c.copy(), m.copy()
    T = len(c)
    for start, length in _runs(m):
        if rng.random() >= s:
            continue
        free = [t for t in range(T - length + 1) if not m[t:t + length].any()]
        if not free:
            continue
        t = int(rng.choice(free))
        a, b = slice(start, start + length), slice(t, t + length)
        c[a], c[b] = c[b].copy(), c[a].copy()
        m[a], m[b] = False, True
    return c, m


def spread_bursts(c, m, s, rng):
    """Move a share (0.75 * s) of each planted hour's excess posts to the two hours on either side."""
    c, m = c.copy(), m.copy()
    T = len(c)
    if not m.any():
        return c, m
    bg = int(np.median(c[~m])) if (~m).any() else 0
    for h in np.flatnonzero(m):
        excess = max(int(c[h]) - bg, 0)
        moved = int(rng.binomial(excess, 0.75 * s)) if excess else 0
        if not moved:
            continue
        nb = [t for t in (h - 2, h - 1, h + 1, h + 2) if 0 <= t < T and not m[t]] or [t for t in (h - 1, h + 1) if 0 <= t < T]
        add = rng.multinomial(moved, np.full(len(nb), 1 / len(nb)))
        c[h] -= moved
        for t, k in zip(nb, add):
            c[t] += k
            if k:
                m[t] = True  # the spread posts still carry the planted pattern
    return c, m


def daily_rhythm(c, m, s, rng, is_bot):
    """Bots only: re-time each background post (outside planted hours) with probability s to follow a human day."""
    if not is_bot:
        return c.copy(), m.copy()
    c, m = c.copy(), m.copy()
    T = len(c)
    free = np.flatnonzero(~m)
    n = int(c[free].sum())
    if n == 0 or not len(free):
        return c, m
    moved = int(rng.binomial(n, s))
    if not moved:
        return c, m
    # take `moved` posts out uniformly from the background, then put them back following a circadian profile
    take = rng.multivariate_hypergeometric(c[free].astype(np.int64), moved)
    c[free] -= take.astype(c.dtype)
    peak = rng.normal(19, 3)
    hod = np.arange(T)[free] % 24
    p = 1 + 0.8 * np.cos(2 * np.pi * (hod - peak) / 24)
    c[free] += rng.multinomial(moved, p / p.sum()).astype(c.dtype)
    return c, m


def match_volume(c, m, s, rng, donor_total):
    """Thin out or top up the week to move its total (in log space) a share s of the way to a real user's total."""
    c = c.copy()
    own = int(c.sum())
    target = int(round(np.exp((1 - s) * np.log1p(own) + s * np.log1p(donor_total)) - 1))
    if target < own:
        c = rng.binomial(c.astype(np.int64), target / max(own, 1)).astype(c.dtype)
    elif target > own:
        p = c.astype(float) + 0.1
        c = c + rng.multinomial(target - own, p / p.sum()).astype(c.dtype)
    return c, m.copy()


def apply_activity_attack(counts, masks, kind, label, attack_key, strength, donors, rng):
    """Return attacked (counts, masks) for every inauthentic account; real users are copied unchanged."""
    counts, masks = counts.copy(), masks.copy()
    if strength == 0:
        return counts, masks
    steps = ["shift_bursts", "spread_bursts", "daily_rhythm", "match_volume"] if attack_key == "activity_combined" else [attack_key]
    totals = counts.sum(1)
    for i in np.flatnonzero(label == 1):
        c, m = counts[i], masks[i]
        for step in steps:
            if step == "shift_bursts":
                c, m = shift_bursts(c, m, strength, rng)
            elif step == "spread_bursts":
                c, m = spread_bursts(c, m, strength, rng)
            elif step == "daily_rhythm":
                c, m = daily_rhythm(c, m, strength, rng, kind[i] == "bot")
            elif step == "match_volume":
                c, m = match_volume(c, m, strength, rng, int(totals[donors[i]]))
        counts[i], masks[i] = c, m
    return counts, masks


# ======================================================================== post text (Phase 4)
DISGUISE = {"giveaway": "g1veaway", "promo": "pr0mo", "crypto": "crypt0", "click": "cl1ck", "winner": "w1nner", "dm": "d.m",
            "idiot": "1diot", "sheep": "sh33p", "liar": "l1ar", "shill": "sh1ll", "clown": "cl0wn", "wake": "wak3"}
SYNONYMS = {"giveaway": "freebie", "promo": "deal", "crypto": "bitcoin", "click": "tap", "winner": "champion", "dm": "inbox",
            "idiot": "fool", "sheep": "followers", "liar": "fraud", "shill": "puppet", "clown": "joker", "wake": "awaken"}

TEXT_ATTACKS = {
    "disguise_markers": Attack("disguise_markers", "Disguise the giveaway words", ("marker words",), "cheap",
                               "Misspell the tell-tale words (g1veaway, sh33p) so a word-level model no longer recognises them."),
    "synonyms": Attack("synonyms", "Swap in synonyms", ("marker words",), "cheap",
                       "Replace the tell-tale words with ordinary synonyms the model never saw in training (freebie, fool)."),
    "padding": Attack("padding", "Pad posts with ordinary words", ("post length",), "cheap",
                      "Add everyday words to every post so the tell-tale word is a smaller part of it."),
}
TEXT_COMBINED = Attack("text_combined", "All text changes", ("marker words", "post length"), "cheap",
                       "Disguise or swap every tell-tale word (half each) and pad every post.")
TEXT_ATTACKS["text_combined"] = TEXT_COMBINED

NEUTRAL_WORDS = None  # filled lazily from anonsynth.text.NEUTRAL


def _neutral():
    global NEUTRAL_WORDS
    if NEUTRAL_WORDS is None:
        from anonsynth.text import NEUTRAL
        NEUTRAL_WORDS = list(NEUTRAL)
    return NEUTRAL_WORDS


def attack_post(tokens, attack_key, s, rng, markers):
    """Return (new tokens, [(original, replacement), ...], number of words added) for one post."""
    toks = list(tokens)
    swaps = []
    if attack_key in ("disguise_markers", "synonyms", "text_combined"):
        for j, t in enumerate(toks):
            if t in markers and rng.random() < s:
                use = attack_key if attack_key != "text_combined" else ("disguise_markers" if rng.random() < 0.5 else "synonyms")
                new = (DISGUISE if use == "disguise_markers" else SYNONYMS)[t]
                swaps.append((t, new))
                toks[j] = new
    added = 0
    if attack_key in ("padding", "text_combined"):
        added = int(round(s * len(tokens)))
        for _ in range(added):
            toks.insert(int(rng.integers(0, len(toks) + 1)), str(rng.choice(_neutral())))
    return toks, swaps, added


FAMILY_ATTACKS = {
    "tabular": (ATTACKS, COMBINED.key),
    "sequences": (ACTIVITY_ATTACKS, ACTIVITY_COMBINED.key),
    "text": (TEXT_ATTACKS, TEXT_COMBINED.key),
}
