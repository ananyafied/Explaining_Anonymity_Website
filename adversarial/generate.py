"""Generate the adversarial dataset.

    python -m adversarial --seed 1 --out adversarial/data

Writes:
  <out>/base/                 a new population in exactly the format of data/synthetic/
                              (#trending removed from every post; nothing attacked yet)
  <out>/<attack>_<strength>/  the files that attack changes, applied to every bot and troll:
                              accounts.csv (account stats), sequences.npz (hourly activity) or
                              posts.csv (text), plus truth/ files saying what was changed
  <out>/manifest.json         seed, strengths, attacks and the list of variants

Account stats (Phase 3), hourly activity and post text (Phase 4) are attacked; each variant folder
holds only the files its attack changes (everything else is in base/). Edges are as generated.
Everything is deterministic given the seed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from anonsynth.config import SynthConfig
from anonsynth.generate import generate as generate_population

from .attacks import (ACTIVITY_ATTACKS, ATTACKS, STRENGTHS, TEXT_ATTACKS, apply_activity_attack, apply_attack,
                      attack_post, changed_features, pick_donors)

DEFAULT_SEED = 1  # the original data uses seed 0; a different seed gives a fresh population


def population_config(seed: int, n_accounts: int = 5000) -> SynthConfig:
    """Same generating rules as the original data, but #trending never appears."""
    return replace(SynthConfig(), n_accounts=n_accounts, seed=seed,
                   spurious_prob_train_inauth=0.0, spurious_prob_test_inauth=0.0, spurious_prob_authentic=0.0)


def variant_name(attack_key: str, strength: str) -> str:
    return f"{attack_key}_{strength}"


def _rng(seed, *path):
    """Independent, reproducible random stream for one (family, attack, strength)."""
    return np.random.default_rng(np.random.SeedSequence([seed, 2026, *path]))


def build_activity(pop, donors, seed):
    """{variant: (counts, masks)} for every activity attack and non-zero strength."""
    kind = pop.truth["kind"].to_numpy()
    label = pop.accounts["label"].to_numpy()
    out = {}
    for ai, key in enumerate(ACTIVITY_ATTACKS):
        for si, (sname, s) in enumerate(STRENGTHS.items()):
            if s == 0:
                continue
            out[variant_name(key, sname)] = apply_activity_attack(pop.seq_counts, pop.seq_masks, kind, label, key, s, donors, _rng(seed, 1, ai, si))
    return out


def build_text(pop, seed):
    """{variant: (posts, posts_truth, text_attack_truth)} for every text attack and non-zero strength."""
    from anonsynth.text import BOT_MARKERS, TROLL_MARKERS
    markers = set(BOT_MARKERS) | set(TROLL_MARKERS)
    fake_acc = set(pop.accounts.loc[pop.accounts["label"] == 1, "account_id"])
    out = {}
    for ai, key in enumerate(TEXT_ATTACKS):
        for si, (sname, s) in enumerate(STRENGTHS.items()):
            if s == 0:
                continue
            rng = _rng(seed, 2, ai, si)
            texts, swaps_col, added_col = [], [], []
            for aid, text in zip(pop.posts["account_id"], pop.posts["text"]):
                toks = text.split()
                if aid in fake_acc:
                    toks, swaps, added = attack_post(toks, key, s, rng, markers)
                else:
                    swaps, added = [], 0
                texts.append(" ".join(toks))
                swaps_col.append("|".join(f"{a}>{b}" for a, b in swaps))
                added_col.append(added)
            posts = pop.posts.assign(text=texts)
            truth = pd.DataFrame({"post_id": pop.posts["post_id"], "account_id": pop.posts["account_id"],
                                  "original_text": pop.posts["text"], "replacements": swaps_col, "words_added": added_col})
            out[variant_name(key, sname)] = (posts, pop.posts_truth.copy(), truth)
    return out


def build(seed: int = DEFAULT_SEED, n_accounts: int = 5000):
    """Returns (population, donors, {variant: (accounts, attack_truth)}) for the account-stat attacks."""
    pop = generate_population(population_config(seed, n_accounts))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2026]))  # independent of the population's streams
    donors = pick_donors(pop.accounts, rng)
    variants = {}
    fake = pop.accounts["label"].to_numpy() == 1
    for key, attack in ATTACKS.items():
        for sname, s in STRENGTHS.items():
            if s == 0:
                continue
            acc = apply_attack(pop.accounts, attack, s, donors)
            truth = pd.DataFrame({
                "account_id": pop.accounts["account_id"],
                "kind": pop.truth["kind"],
                "attacked": fake,
                "donor_id": np.where(fake, pop.accounts["account_id"].to_numpy()[donors], -1),
            })
            for col in changed_features(attack):
                truth[f"original_{col}"] = pop.accounts[col].to_numpy()
            variants[variant_name(key, sname)] = (acc, truth)
    return pop, donors, variants


def write(out_dir, seed: int = DEFAULT_SEED, n_accounts: int = 5000) -> Path:
    out = Path(out_dir)
    pop, donors, variants = build(seed, n_accounts)
    pop.ground_truth["adversarial"] = {
        "seed": seed,
        "note": "#trending is removed from every post. Account stats of bots and trolls are attacked in the variant folders; base/ is unattacked.",
    }
    pop.write(out / "base")
    for name, (acc, truth) in variants.items():
        d = out / name
        (d / "truth").mkdir(parents=True, exist_ok=True)
        acc.to_csv(d / "accounts.csv", index=False)
        truth.to_csv(d / "truth" / "attack_truth.csv", index=False)
    activity = build_activity(pop, donors, seed)
    for name, (counts, masks) in activity.items():
        d = out / name
        (d / "truth").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "sequences.npz", account_id=pop.accounts["account_id"].to_numpy(), counts=counts)
        np.savez_compressed(d / "truth" / "sequence_masks.npz", masks=masks)
    text_variants = build_text(pop, seed)
    for name, (posts, posts_truth, truth) in text_variants.items():
        d = out / name
        (d / "truth").mkdir(parents=True, exist_ok=True)
        posts.to_csv(d / "posts.csv", index=False)
        posts_truth.to_csv(d / "truth" / "posts_truth.csv", index=False)
        truth.to_csv(d / "truth" / "text_attack_truth.csv", index=False)
    manifest = {
        "seed": seed,
        "n_accounts": n_accounts,
        "strengths": STRENGTHS,
        "attacks": {k: {"name": a.name, "features": list(a.features), "changes": list(changed_features(a)), "cost": a.cost, "how": a.how}
                    for k, a in ATTACKS.items()},
        "variants": sorted(variants),
        "activity_attacks": {k: {"name": a.name, "changes": list(a.features), "cost": a.cost, "how": a.how} for k, a in ACTIVITY_ATTACKS.items()},
        "activity_variants": sorted(activity),
        "text_attacks": {k: {"name": a.name, "changes": list(a.features), "cost": a.cost, "how": a.how} for k, a in TEXT_ATTACKS.items()},
        "text_variants": sorted(text_variants),
        "evaluation_split": "test",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-accounts", type=int, default=5000)
    p.add_argument("--out", default="adversarial/data")
    a = p.parse_args()
    out = write(a.out, a.seed, a.n_accounts)
    m = json.loads((out / "manifest.json").read_text())
    n = len(m["variants"]) + len(m["activity_variants"]) + len(m["text_variants"])
    print(f"wrote {out}: base population + {n} attacked variants (seed {a.seed})")


if __name__ == "__main__":
    main()
