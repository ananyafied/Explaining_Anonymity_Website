# Explaining Anonymity: Adversarial Attacks and Science Communication

**Live site: https://ananyafied.github.io/Explaining_Anonymity_Website/**

[![Tests](https://github.com/ananyafied/Explaining_Anonymity_Website/actions/workflows/tests.yml/badge.svg)](https://github.com/ananyafied/Explaining_Anonymity_Website/actions/workflows/tests.yml)
[![Deploy site](https://github.com/ananyafied/Explaining_Anonymity_Website/actions/workflows/pages.yml/badge.svg)](https://github.com/ananyafied/Explaining_Anonymity_Website/actions/workflows/pages.yml)

Ananya Jogalekar · Final project for the Explainable AI course at Duke University. Builds on
[Explaining_Anonymity](https://github.com/ananyafied/Explaining_Anonymity).

![A model page on the Explaining Anonymity site](docs/img/site_model_page.png)

## Research question

**Can we use explainability to understand how state-of-the-art models for detecting anonymous
fake accounts make their decisions, and do those explanations stay trustworthy when the fakes
adapt?**

The project answers this in two phases:

1. **Do the explanations point at the real evidence?** (normal data) We generate synthetic
   accounts where we know exactly what makes an account fake, train models from each family used
   in the literature, explain them with the methods most common for each family, and grade every
   explanation against the answer key on correctness, faithfulness, agreement and stability.
2. **Do the models and their explanations survive when the fakes start hiding?** (adversarial
   data) We build a fresh population whose bots and trolls disguise their account stats, posting
   activity or posts, at mild and strong strength, and re-test every model and its explanations.

The results are published as an interactive website for developers deciding whether to deploy
such a model: pick one of 18 models and see what it relies on, whether its explanations are
correct, how it holds up under attack, and a checklist of what to test before deploying it.

## Grounding in the literature

**Model selection.** The models were chosen from a survey of state-of-the-art work on detecting
bots, trolls and anonymous authors. Each model family below follows a published detector (a
non-exhaustive list of the papers referenced for model selection):

| Model family | Literature anchor |
|---|---|
| Random forest | Davis et al. (2016), BotOrNot/Botometer, a random forest over account features |
| XGBoost | Shevtsov et al. (2022), an XGBoost bot detector explained with SHAP |
| SVM | Chun et al. (2019), an SVM troll classifier on IRA tweets |
| LSTM | Kudugunta & Ferrara (2018), a contextual LSTM for bot detection |
| BERT-style text | Fabien et al. (2020), BertAA, a fine-tuned BERT for authorship |
| Relational GCN | Feng et al. (2021), BotRGCN; TwiBot-22 benchmarks this family |
| LSTM autoencoder | MulBot (Mannocci et al. 2022) and RTbust (Mazza et al. 2019), time-series autoencoders |
| Text-graph fusion | Lei et al. (2023), BIC, a text-graph interaction model for bot detection |

The graph models (R-GCN and relatives) and the text-graph fusion model were trained and explained
in phase 1 but are not on the website, because the follower graph was not attacked in phase 2.

**Explanation methods and evaluation.** The explanations use standard feature-attribution methods:
SHAP (Lundberg & Lee, 2017), LIME (Ribeiro, Singh & Guestrin, 2016), Integrated Gradients
(Sundararajan, Taly & Yan, 2017), Grad-CAM (Selvaraju et al., 2017) and, in phase 1, GNNExplainer
(Ying et al., 2019). They are scored on criteria common in the literature on evaluating
explanations: correctness against a known ground truth, faithfulness (a deletion test: does
removing the top-ranked inputs change the score?), agreement between methods (explanations of the
same model often disagree: Krishna et al., 2022), and stability.

**Responsible AI themes.** The project connects to course concepts on interpretability (local and
global feature attribution, and whether attributions can be trusted), shortcut learning (the
`#trending` trap, after Geirhos et al., 2020), the fragility of post-hoc explanations (Slack et
al., 2020), and adversarial evasion of bot detectors (Cresci et al., 2019; Cresci, 2020, on the
arms race between detectors and bot operators).

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

## Evidence and evaluation

- **Controlled experiments with ground truth.** Every explanation is scored against the planted
  answer key; every model is re-tested on the same fresh population under each attack at three
  strengths (a dose-response), with the unattacked population as the control.
- **Quantitative criteria with uncertainty.** AUC with 95% bootstrap intervals, plus correctness,
  faithfulness (deletion test), agreement and stability for every explanation method.
- **Verification.** 66 automated tests (run on every push), a check that every saved model
  reproduces its predictions, and checks that each attack changes only what it should.
- **Iterative development.** The work was built in reviewed phases (setup, normal-data results,
  website, account-stat attacks, activity and text attacks, finishing), each deployed and checked
  before the next. Findings fed back into the design: for example, the "ignores timing" weakness
  found on normal data was confirmed by the activity attacks, and a sign error in the stored LIME
  results was found and corrected during website development.

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

## Responsible AI considerations

- **Uncertainty.** AUCs are reported with 95% bootstrap intervals (shown on each model page), and
  explanation scores are averages over samples of 150 accounts (60 posts for text), so a model
  just above or below a verdict threshold should be read as borderline. Findings show how these
  models *can* fail, not estimates of real-world failure rates.
- **Failure modes the project surfaces.** Models that rely on the right features but miss
  interactions; models that count posts instead of reading timing; shortcut learning (`#trending`);
  explanation methods that disagree or change between runs; models that collapse, or even invert,
  under cheap disguises while their explanations barely change; and decision thresholds that flag
  most real users or stop working once a shortcut disappears.
- **Stakeholders.** Developers and trust-and-safety teams deploying detectors (the site's main
  audience); moderators who act on flags and their explanations; **real users, who bear the cost
  of false positives**, including people who are legitimately anonymous or pseudonymous
  (activists, whistleblowers, people protecting their safety), for whom being flagged as fake is a
  harm; and researchers comparing detectors.
- **Anonymity is not the target.** The models detect *inauthentic behaviour* (automation,
  coordination, spam), not anonymity itself. A deployment that treats "anonymous" as "suspicious"
  would harm the people anonymity protects; the checklist's threshold and human-review items exist
  for this reason.
- **Misuse risks.** The attack catalogue could in principle guide evasion. The attacks are
  deliberately generic and black-box (copying what real users visibly do), they are already
  standard in the bot-detection literature, they run only on synthetic data, and the project
  publishes them to help defenders test for them before attackers do. A second risk is
  over-trust: a "good" verdict is not a guarantee of safety, which the site states and the
  attack section demonstrates.
- **Deployment considerations.** Every model page ends with a checklist generated from its own
  evidence: test on later data, set thresholds from the cost of wrongly flagging a real user, keep
  an audit sample, red-team with cheap disguises, and do not rely on one easily faked behaviour.
- **Privacy.** No real accounts or personal data are used anywhere; all data is synthetic.

## Why it matters

Explainability is often offered as the safeguard for high-stakes classifiers: if a model can say
why it flagged an account, people can check it. This project tests that assumption end to end and
finds it weaker than hoped. On normal data, explanations often point at the right evidence, but
the two methods disagree and miss interactions. Under attack, every model falls to cheap
disguises, and for the account-stat models only 3–6% of SHAP's credit moves while they fail, so
explanations alone would not warn a team that its detector is being evaded.

The contribution is threefold:

1. **An evaluation**: a synthetic benchmark with an answer key that grades explanations of
   detectors across three input types (five, including the graph and multimodal models in phase 1).
2. **A dataset**: an adversarial population with graded, cost-labelled disguises, reproducible from
   a seed, that measures how both accuracy and explanations degrade.
3. **An application**: a website that turns these results into per-model verdicts, vulnerabilities
   and pre-deployment checklists through fixed, tested rules, as a prototype step in a deployment
   pipeline.

## Future work

- **Real data.** Repeat the evaluation on a labelled benchmark such as TwiBot-22, using the
  synthetic answer key to validate the method first.
- **Graph and combined attacks.** Attack the follower graph (inserting accounts, diluting troll
  rings) and disguise all inputs at once, as a real operator would.
- **Adaptive attackers.** Test white-box attackers that query the model, and models retrained on
  adversarial data (adversarial training), to see whether explanations recover.
- **Bring your own model.** Let teams upload a model and data so the site's checks run as a real
  step in a deployment pipeline (for example in continuous integration).
- **Monitoring with explanations.** Explanations barely moved during the account-stat attacks;
  test whether other signals (score distributions, explanation drift on audit samples) detect
  evasion earlier.
- **A user study.** Ask developers and moderators whether the verdicts and checklists change their
  decisions, and which explanations they trust.
- **Character-aware text models.** Disguised and synonym-swapped words looked identical to the
  word-level models; models that read characters may resist disguises.

## References

- Chun, et al. (2019). An SVM troll classifier on IRA tweets.
- Cresci, S. (2020). A decade of social bot detection. *Communications of the ACM*.
- Cresci, S., Petrocchi, M., Spognardi, A., & Tognazzi, S. (2019). Better safe than sorry: an adversarial approach to improve social bot detection. *ACM WebSci*.
- Davis, C. A., Varol, O., Ferrara, E., Flammini, A., & Menczer, F. (2016). BotOrNot: A system to evaluate social bots. *WWW Companion*.
- Fabien, M., et al. (2020). BertAA: BERT fine-tuning for authorship attribution.
- Feng, S., et al. (2021). BotRGCN: Twitter bot detection with relational graph convolutional networks.
- Feng, S., et al. (2022). TwiBot-22: Towards graph-based Twitter bot detection. *NeurIPS Datasets and Benchmarks*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*.
- Krishna, S., et al. (2022). The disagreement problem in explainable machine learning: a practitioner's perspective.
- Kudugunta, S., & Ferrara, E. (2018). Deep neural networks for bot detection. *Information Sciences*.
- Lei, Z., et al. (2023). BIC: Twitter bot detection with text-graph interaction and semantic consistency. *ACL*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Mannocci, L., et al. (2022). MulBot: Unsupervised bot detection based on multivariate time series.
- Mazza, M., et al. (2019). RTbust: Exploiting temporal patterns for botnet detection on Twitter. *ACM WebSci*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Shevtsov, A., et al. (2022). An XGBoost bot detector explained with SHAP.
- Slack, D., et al. (2020). Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. *AIES*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Ying, R., et al. (2019). GNNExplainer: Generating explanations for graph neural networks. *NeurIPS*.

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
| `tests/` | Tests for the generators, the metrics and the site's rules (run on every push by `.github/workflows/tests.yml`) |

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
