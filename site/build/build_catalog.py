"""Write site/public/data/catalog.json: the models the website shows, grouped by family.

Run from anywhere:  python site/build/build_catalog.py
Needs only the standard library (reads outputs/step1_fit/registry.json and metrics.csv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from catalog import EXCLUDED, FAMILIES, FAMILY_INFO, METHOD_INFO, MODELS

ROOT = Path(__file__).resolve().parents[2]
STEP1 = ROOT / "outputs" / "step1_fit"
OUT = ROOT / "site" / "public" / "data" / "catalog.json"


def main() -> None:
    registry = json.loads((STEP1 / "registry.json").read_text())
    with open(STEP1 / "metrics.csv", newline="") as f:
        auc = {r["model"]: float(r["test_roc_auc"]) for r in csv.DictReader(f)}
    catalog = {
        "families": [{"key": k, "name": v[0], "about": FAMILY_INFO[k]} for k, v in FAMILIES.items()],
        "methods_info": METHOD_INFO,
        "models": [
            {
                "key": m.key,
                "name": m.name,
                "family": m.family,
                "kind": m.kind,
                "input": registry[m.key]["input_view"],
                "methods": list(m.methods),
                "aliases": list(m.aliases),
                "test_auc": round(auc[m.key], 4),
            }
            for m in MODELS
        ],
        "excluded": EXCLUDED,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(catalog['models'])} models")


if __name__ == "__main__":
    main()
