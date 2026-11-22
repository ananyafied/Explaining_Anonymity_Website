# Explaining Anonymity: Adversarial Attacks and Science Communication

**Live site: https://ananyafied.github.io/Explaining_Anonymity_Website/**

Ananya Jogalekar · Final project for the Explainable AI course at Duke University. Builds on
[Explaining_Anonymity](https://github.com/ananyafied/Explaining_Anonymity).

![A model page on the Explaining Anonymity site](docs/img/site_model_page.png)

## Purpose

Models that detect fake, bot and troll accounts are usually judged on accuracy, and their
explanations (SHAP, LIME and similar) are usually inspected one plot at a time. This project asks
two questions a developer should answer before deploying such a model:

1. **Do its explanations point at the real evidence?** We generate synthetic accounts where we know
   exactly what makes an account fake, so every explanation can be graded right or wrong.
2. **Does it survive when the fakes start hiding?** We build an adversarial population whose bots
   and trolls disguise their behaviour, and re-test every model and its explanations.

The answers are published as an interactive website: pick one of 18 models and see what it relies
on, whether its explanations are correct, how it holds up under attack, and a checklist of what to
test before deploying it.

## Viewing the project

- **Website:** open the live site above. No installation needed. The "About" page explains the
  method in plain language.
- **Locally, without installing anything:** the site is static. From the repository root run
  `python3 -m http.server -d site/public 8000` and open http://localhost:8000.
- **Write-ups:** [docs/](docs/), listed under [Documentation](#documentation) below.
- **Poster:** [poster/poster.pdf](poster/poster.pdf) (36 × 24 in).

## Reproducing the project

Tested with **Python 3.10** on macOS (Apple silicon). Every step is deterministic from fixed seeds.

```bash
# 0. environment (pinned versions; needed to reload the saved models exactly)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements-site.txt

# 1. data (not committed; regenerated exactly from seeds)
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # original data, a few seconds
python -m adversarial                                                 # adversarial data, ~10 seconds

# 2. checks
python -m pytest                       # 66 tests: generators, metrics, labelling rules
python site/build/verify_models.py     # all 18 models reproduce their saved predictions

# 3. adversarial results (already committed in outputs/step3_adversarial/; re-run only to regenerate)
python -m adversarial.evaluate_tabular     # ~10 minutes
python -m adversarial.evaluate_sequences   # ~2 minutes
python -m adversarial.evaluate_text        # ~5 minutes

# 4. the website's data (seconds; reads saved results, loads no models)
python site/build/build_models.py
python -m http.server -d site/public 8000  # preview at http://localhost:8000
```

Pushing changes under `site/public/` to `main` redeploys the live site
(`.github/workflows/pages.yml`).

**Optional:**
- *Re-run the step 2 explainability notebooks:* `pip install -r requirements-xai.txt`, then run
  `notebooks/02` to `07` in order. Each has a `QUICK = True` switch for a few-minute smoke test.
- *Rebuild the poster:* `pip install -r requirements-poster.txt` (Google Chrome must be installed),
  then `python poster/build_poster.py` (PDF) and `python poster/build_poster_pptx.py` (PowerPoint).

## What is included

| Item | Where | Committed? |
|---|---|---|
| Synthetic data generator | `anonsynth/` | yes (data regenerated with `python -m anonsynth`, seed 0, ~6 MB) |
| Adversarial data generator and evaluation | `adversarial/` | yes (data regenerated with `python -m adversarial`, seed 1, ~59 MB) |
| Trained models (28) and their predictions | `outputs/step1_fit/` | yes (47 MB) |
| Explainability results (step 2) | `outputs/step2_explain/`, `notebooks/` | yes (16 MB; notebook outputs cleared) |
| Adversarial results (step 3) | `outputs/step3_adversarial/` | yes (8.5 MB) |
| Model loaders and explanation metrics | `xai_eval/` | yes |
| Website and the scripts that build its data | `site/` | yes |
| Tests | `tests/` | yes (66) |
| Poster and its build scripts | `poster/` | yes |
| Pinned dependencies | `requirements-site.txt` (all steps), `requirements-xai.txt` (notebooks), `requirements-poster.txt` (poster) | yes |

The datasets are generated rather than committed because they are rebuilt exactly from their seeds
in seconds, and the tests check that they are. Everything else needed to reproduce the results
(the saved models, every result file, and the code) is in the repository.

## How it works

1. **Synthetic data with an answer key** (`anonsynth/`): 5,000 accounts, about 30% bots or trolls.
   Account stats follow a known formula, bots and trolls post in planted burst patterns, and their
   posts carry planted giveaway words. A trap hashtag, `#trending`, signals fakes only in training.
   [Details](docs/synthetic_data.md).
2. **Models** (`outputs/step1_fit/`): 28 detectors across account stats, hourly activity, post
   text, the follower graph and a multimodal model.
3. **Explanations** (`notebooks/`, `xai_eval/`, `outputs/step2_explain/`): each model is explained
   with the two methods most common for its type and scored on correctness, faithfulness,
   agreement and stability. [The plan](docs/step2_plan.md).
4. **Adversarial dataset** (`adversarial/`, `outputs/step3_adversarial/`): a fresh population
   whose bots and trolls disguise their account stats, hourly activity or posts, at mild and strong
   strength. [How it was built](docs/adversarial_dataset.md) ·
   [the attacks](docs/adversarial_attacks.md) · [results by model](docs/adversarial_results.md).
5. **Website** (`site/`): Python in `site/build/` turns the saved results into JSON through fixed,
   tested rules ([docs/site_rules.md](docs/site_rules.md)); `site/public/` is plain HTML, CSS and
   JavaScript served by GitHub Pages.

## Major design decisions

- **Synthetic data with a known answer key.** Real fake-account data has no ground truth for *why*
  an account is fake, so explanations could only be judged as plausible. Synthetic data lets every
  explanation be scored as right or wrong.
- **Each model keeps the two explanation methods most used for its type** (SHAP and LIME for the
  account-stat and text models; Integrated Gradients with GradientSHAP or Grad-CAM for the activity
  models) rather than forcing SHAP and LIME on everything, where they would be slow, noisy and
  untypical of practice.
- **Verdicts come from fixed, published rules, never hand-written notes.** A model is "good" only
  if it is accurate (within 0.06 AUC of the best achievable or best in its family), holds up from
  validation to test, and its explanations find the planted signal. Every threshold is documented
  and covered by tests that fail if the docs and code disagree.
- **A fresh adversarial population, not a perturbed test set.** Bots and trolls are disguised; real
  users and labels never change, `#trending` is removed, and models are used exactly as trained
  (original scaling, no refitting), as a deployed model would be. Attacks are black-box
  ("mimicry": copy what real users visibly do) and labelled by their cost to an operator.
- **Only attacked models are shown.** The 7 graph models and the multimodal model were trained and
  explained but never attacked (graph attacks were left out of scope), so they are not on the
  site. Belief propagation (inverted scores) and SVD + logistic regression (identical to PCA +
  logistic regression) were dropped earlier. Their results remain in `outputs/`.
- **A static website with precomputed results.** All results are deterministic, so the site needs
  no server: it is fast, free to host on GitHub Pages and cannot break during a demo. The Python
  that builds its data is the "backend" and lives in the repository.
- **Pinned versions.** The saved models are pickles; `requirements-site.txt` pins the versions that
  `site/build/verify_models.py` confirms reproduce every model's saved predictions.

## Known limitations

- **The data is synthetic.** Real accounts are messier, and real operators adapt in ways a
  generator does not. The findings show how these models and explanations *can* fail, not how
  often they would on a real platform.
- **The attacker cannot see the models** (black-box mimicry). An attacker who could query the
  models would do at least as well.
- **Each input type is attacked separately.** A real operator would disguise account stats,
  activity and posts at once. The follower graph was not attacked at all.
- **Explanations are scored on samples** of 150 accounts (60 posts for text), so small differences
  between models are not meaningful.
- **Decision thresholds come from step 1.** The anomaly detectors' thresholds flag most real users,
  and the text models' thresholds were tuned while `#trending` was present; the site flags both.
- **The model-fitting notebook is not included.** The network definitions it used are in
  `xai_eval/step1_models.py`, and every saved model is verified to reproduce its predictions, but
  the models cannot be retrained from this repository.
- **doc2vec is not exactly reproducible.** Its inference uses a per-process random hash; it is
  checked by rank correlation (above 0.95) instead of exact match.
- **Tested on macOS with Python 3.10 only.** On macOS, LightGBM must be imported before PyTorch
  (the scripts do this) or loading the LightGBM model crashes.
- **The site covers these 18 models only.** Plugging in your own model or data is the next step,
  not a current feature.
- **Known issue in the step 2 data:** the stored LIME weights for the account-stat models (and for
  node2vec and DeepWalk) have the wrong sign; their sizes are right and no step 2 score is
  affected. The site corrects the sign where it shows direction.
  [Details](docs/site_rules.md#6-known-issue-in-the-step-2-data-lime-signs-account-stats).

## Repository layout

| Folder | What it holds |
|---|---|
| `anonsynth/` | Synthetic data generator (`python -m anonsynth`) |
| `xai_eval/` | Model loaders and explanation metrics shared by the notebooks and scripts |
| `notebooks/` | Step 2 explainability notebooks (outputs cleared) |
| `adversarial/` | Adversarial data generator and evaluation scripts ([technical notes](adversarial/README.md)) |
| `outputs/` | Saved results: `step1_fit/` (models), `step2_explain/`, `step3_adversarial/` |
| `site/` | The website: `public/` is served, `build/` generates its data ([notes](site/README.md)) |
| `poster/` | The poster (PDF) and the scripts that build it |
| `docs/` | Write-ups and the labelling rules |
| `tests/` | Tests for the generators, the metrics and the site's rules |

## Documentation

| Document | What it covers |
|---|---|
| [docs/synthetic_data.md](docs/synthetic_data.md) | How the synthetic data and its answer key are generated |
| [docs/step2_plan.md](docs/step2_plan.md) | How the explanations were computed and scored |
| [docs/adversarial_dataset.md](docs/adversarial_dataset.md) | How the adversarial dataset was built and tested |
| [docs/adversarial_attacks.md](docs/adversarial_attacks.md) | Every attack, its strengths and its cost |
| [docs/adversarial_results.md](docs/adversarial_results.md) | A short result per attacked model |
| [docs/site_rules.md](docs/site_rules.md) | The rules behind every verdict, strength, weakness and checklist item |
| [adversarial/README.md](adversarial/README.md) | Technical notes on the adversarial code and files |
| [site/README.md](site/README.md) | How the website and its data build work |
