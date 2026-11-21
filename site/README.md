# Website

The *Explaining Anonymity* website. It is static: `public/` is what GitHub Pages serves, and
`.github/workflows/pages.yml` deploys it on every push to `main` that touches `site/public/`.

`build/` holds the Python that produces the site's data:

| Script | What it does |
|---|---|
| `build/catalog.py` | The 26 models the site shows (names, aliases, family, explanation methods) and the excluded ones |
| `build/build_catalog.py` | Writes `public/data/catalog.json` |
| `build/verify_models.py` | Reloads all 26 models and checks they reproduce their step-1 predictions |
| `build/facts.py` | Collects each model's numbers from `outputs/step1_fit` and `outputs/step2_explain` |
| `build/rules.py` | Turns those numbers into a verdict, strengths, weaknesses and a checklist ([rules](../docs/site_rules.md), tested in `tests/test_site_rules.py`) |
| `build/views.py` | Family-specific chart data: importance from each method, the answer key, examples |
| `build/build_models.py` | Writes `public/data/models/<model>.json` for all 26 models and refreshes `catalog.json` |

```bash
pip install -r requirements-site.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic
python site/build/verify_models.py     # expect 26/26
python site/build/build_models.py      # also rebuilds catalog.json; no models are loaded
```

To preview locally: `python -m http.server -d site/public 8000` and open http://localhost:8000.
