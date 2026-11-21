"""Generate the adversarial dataset.

    python -m adversarial --seed 1 --out adversarial/data

Writes:
  <out>/base/                 a new population in exactly the format of data/synthetic/
                              (#trending removed from every post; nothing attacked yet)
  <out>/<attack>_<strength>/  accounts.csv with that attack applied to every bot and troll,
                              plus truth/attack_truth.csv (what was changed, from what)
  <out>/manifest.json         seed, strengths, attacks and the list of variants

Only account stats are attacked so far (Phase 3); posts, edges and hourly activity in base/ are
as generated. Everything is deterministic given the seed.
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

from .attacks import ATTACKS, STRENGTHS, apply_attack, changed_features, pick_donors

DEFAULT_SEED = 1  # the original data uses seed 0; a different seed gives a fresh population


def population_config(seed: int, n_accounts: int = 5000) -> SynthConfig:
    """Same generating rules as the original data, but #trending never appears."""
    return replace(SynthConfig(), n_accounts=n_accounts, seed=seed,
                   spurious_prob_train_inauth=0.0, spurious_prob_test_inauth=0.0, spurious_prob_authentic=0.0)


def variant_name(attack_key: str, strength: str) -> str:
    return f"{attack_key}_{strength}"


def build(seed: int = DEFAULT_SEED, n_accounts: int = 5000):
    """Returns (population, donors, {variant: (accounts, attack_truth)})."""
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
    manifest = {
        "seed": seed,
        "n_accounts": n_accounts,
        "strengths": STRENGTHS,
        "attacks": {k: {"name": a.name, "features": list(a.features), "changes": list(changed_features(a)), "cost": a.cost, "how": a.how}
                    for k, a in ATTACKS.items()},
        "variants": sorted(variants),
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
    print(f"wrote {out}: base population + {len(m['variants'])} attacked variants (seed {a.seed})")


if __name__ == "__main__":
    main()
