# Website rules: verdicts, strengths, weaknesses and the checklist

The website labels every model **good** or **weak** and lists its strengths and weaknesses. Nothing
is written by hand: fixed rules in [`site/build/rules.py`](../site/build/rules.py) turn each model's
numbers into those labels. This page is the human-readable version of those rules.
`tests/test_site_rules.py` checks that the thresholds below match the code.

Inputs come from step 1 (`outputs/step1_fit/metrics.csv`: test, validation and train AUC) and step 2
(`outputs/step2_explain/<family>/`: the explanation scorecards and probes).

## 1. Verdict: good or weak

A model is **good** only if it passes all three checks. Otherwise it is **weak**.

| Check | Passes when |
|---|---|
| **Accuracy** | Test AUC is at least the family's reference AUC minus `auc_margin`. |
| **Generalises** | Validation AUC minus test AUC is at most `val_test_gap`. A bigger drop means the model learned something that stopped working on new data (in this dataset, mostly the `#trending` shortcut). |
| **Right reasons** | The explanations find the planted signal (family-specific, below). |

### Reference AUC per family

| Family | Reference |
|---|---|
| Account stats | The **best achievable** AUC on the test split: the AUC of the true probability from the generating formula, **0.873**. No model can beat it with these 12 inputs. |
| Hourly activity, post text, follower graph | The best test AUC of any model in the same family. The true ceiling is unknown for these inputs. |
| Multimodal | The best test AUC of any model that reads one of its inputs (account stats, text or graph). |

### "Right reasons" per family

For families with two explanation methods scored on the same measure, the **better** of the two
methods is used, and a separate weakness flags a method that fails (see section 2).

| Family | Measure | Passes when |
|---|---|---|
| Account stats | SHAP's share of credit on the 6 informative features (the 5 linear or threshold features and both inputs of the interaction) | at least `informative_share` |
| Hourly activity | AUC of the attribution for picking out the planted burst hours of bots and trolls (0.5 = chance) | at least `localisation_auc` |
| Post text | Share of credit on planted marker words, in posts that contain one | at least `marker_mass` |
| Graph neural networks | AUC of edge importance for picking out a troll's ring edges (0.5 = chance) | at least `localisation_auc` |
| Node2vec, DeepWalk | Cannot be checked: explanations are about embedding dimensions, which have no meaning | never passes |
| Label spreading | Share of a troll's top-3 seed labels that come from its own ring | at least `seed_ring_top3` |
| Multimodal | Block Shapley values follow the planted signal: text block vs marker-word count and account-stats block vs the true formula (rank correlations), graph block vs ring membership (AUC) | at least `block_follow`, `block_follow` and `block_ring_auc` |

## 2. Strengths and weaknesses

Each line on a model page comes from one of these rules. "Score standard deviations" means the change
in the model's score divided by the score's standard deviation, so probabilities and anomaly scores
are comparable.

**Accuracy**
- *Accurate* (strength): passes the accuracy check.
- *Less accurate than it could be* / *Misses many fakes* (weakness): fails it, by less than or more than 0.10.
- *Barely better than guessing* (weakness): test AUC below `near_chance_auc`.
- *Learned something that did not carry over* (weakness): fails the generalisation check.
- *Memorises its training data* (weakness): train AUC minus test AUC above `train_test_gap`. Not shown when the generalisation weakness already is, and never for propagation models, which are seeded with training labels.
- *Scores how unusual an account is* (weakness): every anomaly detector.

**Account stats**
- *Relies on the right features* / *Only X% of credit…*: the right-reasons check.
- Credit called out when a group gets at least `group_share` of SHAP credit: the proxy `posts_per_week`, the decoys (`follower_following_ratio`, `hashtag_rate`) or the noise features.
- *Underplays the interaction*: the SHAP credit on `burstiness` + `account_age_days` together is below `interaction_fraction` of the interaction term's true share of the signal (19%). SHAP splits an interaction between its inputs, so the pair's combined credit is compared with the term's own share.

**Hourly activity**
- *Points at the planted burst hours* / *does not*: the right-reasons check.
- *Method misses the bursts*: one method passes but the other is below `localisation_auc`.
- *Ignores timing*: shuffling each account's 168 hours costs less than `timing_drop` AUC, so the model counts busy hours rather than reading the rhythm.

**Post text**
- *Puts X% of credit on the marker words* / *only X%*: the right-reasons check.
- *Leans on #trending*: deleting `#trending` from a post moves the score by at least `shortcut_effect` score standard deviations.

**Follower graph**
- *Points at a troll's ring edges* / *barely finds them*: the right-reasons check for the graph neural networks.
- *Acts largely as a degree counter*: rank correlation between the score and the account's number of connections of at least `degree_corr`.
- *Explanations are about embedding dimensions* (node2vec, DeepWalk), with the probe results.
- Label spreading: the share of top seed labels from the troll's own ring, and the fact that its explanation is exact.

**Multimodal**
- Text and account-stats evidence follow the planted signal (rank correlation at least `block_follow`).
- *Graph evidence does not single out troll rings*: AUC below `block_ring_auc`.
- *Mostly a text model*: text alone is within 0.10 AUC of the full model.

**Explanation quality (every family)**
- *The two methods agree* / *disagree*: rank agreement of at least `agree_high` / below `agree_low`. Not used for label spreading, where the two methods agree by construction.
- *Faithful* (strength): deleting the top-ranked inputs moves the score at least `faithful_high` score standard deviations more than deleting random inputs.
- *Not faithful* (weakness): below `faithful_low`.
- *Gives a different answer when re-run*: rank agreement between two runs with different random seeds below `repeat_low`.
- *Unstable*: rank agreement after a tiny change to the input below `stable_low`.

## 3. Thresholds

| Name | Value | Meaning |
|---|---|---|
| `auc_margin` | 0.06 | Accuracy passes within this much of the reference AUC |
| `near_chance_auc` | 0.6 | Below this, "barely better than guessing" |
| `val_test_gap` | 0.05 | Largest allowed drop from validation to test AUC |
| `train_test_gap` | 0.1 | Larger gaps between train and test AUC count as memorising |
| `informative_share` | 0.75 | Account stats: SHAP credit on the 6 informative features |
| `group_share` | 0.1 | Account stats: credit on the proxy, decoys or noise worth calling out |
| `interaction_fraction` | 0.5 | Account stats: interaction credit below this fraction of its true share is "underplayed" |
| `localisation_auc` | 0.7 | Activity and graph: AUC for finding planted hours or ring edges |
| `marker_mass` | 0.5 | Text: share of credit on marker words |
| `shortcut_effect` | 0.5 | Text: score standard deviations moved by deleting `#trending` |
| `timing_drop` | 0.02 | Activity: AUC lost by shuffling hours, below which the model ignores timing |
| `degree_corr` | 0.7 | Graph: rank correlation with degree for "degree counter" |
| `seed_ring_top3` | 0.75 | Label spreading: top-3 seeds from the troll's own ring |
| `block_follow` | 0.5 | Multimodal: rank correlation of a block with its planted signal |
| `block_ring_auc` | 0.7 | Multimodal: graph block AUC for ring membership |
| `agree_high` | 0.7 | The two methods agree |
| `agree_low` | 0.4 | The two methods disagree |
| `faithful_high` | 0.5 | Faithful (score standard deviations) |
| `faithful_low` | 0.2 | Not faithful (score standard deviations) |
| `repeat_low` | 0.8 | Re-runs disagree |
| `stable_low` | 0.7 | Unstable under tiny input changes |
| `attack_collapse` | 0.15 | Under attack: AUC lost to the strong combined attack that counts as collapsing |
| `attack_holds` | 0.05 | Under attack: AUC lost that still counts as holding up |
| `single_attack_drop` | 0.1 | Under attack: AUC lost to one change (strong) that counts as "one change is enough" |
| `importance_shift` | 0.25 | Under attack: share of SHAP credit that moves between no attack and strong |
| `flagged_real_high` | 0.5 | Under attack: share of real users the model's own threshold flags, above which "fakes caught" means little |
| `caught_low` | 0.5 | Under attack: share of fakes the model's threshold catches with no attack, below which the threshold is out of date |
| `inverted_auc` | 0.4 | Under attack: strong combined attack AUC below which fakes look more genuine than real users |
| `trap_gain` | 0.03 | Under attack (text): AUC gained on fresh posts without `#trending`, compared with the original test set |

## 4. Under attack (adversarial dataset)

Each model with adversarial results gets its own list, from
[`adversarial/`](../adversarial/README.md): bots and trolls disguise themselves at mild or strong
strength, one change at a time or all together (the **combined** attack). Account stats copy real
users' values; hourly activity has its bursts moved, spread out, re-timed or thinned to a real
user's volume; posts have their marker words disguised or swapped and are padded with ordinary
words. AUC drops are measured against the same fresh population with no attack.

- *Barely better than guessing even before the attack* (weakness): AUC on the fresh, unattacked population below `near_chance_auc`. Replaces the three rules below and the two "single change" strengths, since such a model has little to lose.
- *Collapses* (weakness): the strong combined attack costs at least `attack_collapse` AUC.
- *Holds up* (strength): it costs at most `attack_holds`. In between: *weakened* (weakness).
- *One cheap change is enough* (weakness): a single cheap attack at strong strength costs at least `single_attack_drop` AUC.
- *Only an expensive change hurts much* (strength): no cheap attack reaches `single_attack_drop`, but buying aged accounts does.
- *No single change hurts much* (strength): no single attack reaches `single_attack_drop`.
- *Only volume matters to it* (weakness, hourly activity): moving bursts and faking a daily rhythm each cost less than `timing_drop` AUC, while matching a real user's weekly volume costs at least `single_attack_drop`.
- *Its most important features are cheap to fake* (weakness): at least two of SHAP's top three features (no attack) are ones a cheap attack changes.
- *A half-hearted attack fools it more than a full one* (weakness): the mild combined attack pushes AUC below 0.5 and below the strong attack's AUC. Fakes moved halfway toward real users are more typical than real users, which fools detectors of unusual accounts.
- *Its threshold is out of date* (weakness): with no attack, the model's own threshold catches fewer than `caught_low` of fakes (not shown when the threshold already flags most real users).
- *Better once #trending is gone* (strength, text): AUC on the fresh, unattacked population is at least `trap_gain` above the original test AUC.
- *The attack turns it upside down* (weakness): the strong combined attack pushes AUC below `inverted_auc`, so disguised fakes score as more genuine than real users.
- *Its threshold flags most real users* (weakness): the model's own decision threshold (from step 1) flags at least `flagged_real_high` of real users, so "fakes caught" is not informative.
- *Explanations change visibly under attack* (strength): at least `importance_shift` of SHAP's credit moves between no attack and the strong combined attack, so monitoring explanations could flag it.
- *Explanations barely change while it fails* (weakness): less than that moves, although the model collapses.

Each of these weaknesses adds an attack test to the model's checklist (`ATTACK_CHECKLIST` in `rules.py`).

## 5. Pre-deployment checklist

Every model page starts with four general checks: test on later data, choose the threshold on
held-out data, keep a fixed audit sample, and test with cheaply faked inputs changed. Each weakness
then adds one specific check (the `TRIGGERED_CHECKLIST` table in `rules.py`), shown with the weakness
that triggered it.

## 6. Known issue in the step 2 data: LIME signs (account stats)

Notebook 02 ran LIME in regression mode and read `next(iter(e.local_exp.values()))`. In regression
mode LIME stores the explanation under label 1 and a sign-flipped copy under label 0, and that
expression picks label 0. Every stored account-stat LIME weight in
`outputs/step2_explain/tabular/attributions_*_LIME.csv` therefore has the right size but the
**wrong sign**. Notebook 05 reads LIME the same way for node2vec and DeepWalk.

None of the step 2 scores are affected: correctness, faithfulness, agreement and stability all use
absolute values or rankings by size. The website only shows signs in the per-account examples, and
`site/build/views.py` flips the tabular LIME sign back before drawing them (tested in
`tests/test_site_views.py`). The stored files are left as they were.
