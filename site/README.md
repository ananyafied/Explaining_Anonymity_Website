# Website

The *Explaining Anonymity* website. It is static: `public/` is what GitHub Pages serves, and
`.github/workflows/pages.yml` deploys it on every push to `main` that touches `site/public/`.

`build/` holds the Python that produces the site's data:

| Script | What it does |
|---|---|
| `build/catalog.py` | The 26 models the site shows (names, aliases, family, explanation methods) and the excluded ones |
| `build/build_catalog.py` | Writes `public/data/catalog.json` |
| `build/verify_models.py` | Reloads all 26 models and checks they reproduce their step-1 predictions |

```bash
pip install -r requirements-site.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic
python site/build/verify_models.py     # expect 26/26
python site/build/build_catalog.py
```

To preview locally: `python -m http.server -d site/public 8000` and open http://localhost:8000.
