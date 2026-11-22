# Results under attack, model by model

*Part 3 of 3. See also [how the dataset was built](adversarial_dataset.md) and [the attacks](adversarial_attacks.md).*

A short summary per model. The full numbers, charts and checklists are on each model's page of
the [website](https://ananyafied.github.io/Explaining_Anonymity_Website/), in the "Under attack"
section. "AUC" is the chance that a random fake scores above a random real user: 0.5 is guessing,
1.0 is perfect, and below 0.5 means the model ranks fakes as *more* genuine than real users. AUCs
are rounded; "strong" means the combined attack at full strength.

## Overall picture

- **Every model that was attacked can be defeated by cheap changes.** No model held up under the
  strong combined attack of its family.
- **One change is usually enough.** Each family has a single cheap behaviour carrying most of the
  signal: posting rate or replies for account stats, weekly volume for activity, the giveaway word
  for text.
- **Explanations rarely warn you.** For the account-stat models, only 3–6% of SHAP's credit moves
  while the models fail, so watching explanations would not reveal the attack. For the activity
  and text models the credit follows the signal away, which describes the failure but would not
  prevent it.
- **Some decision thresholds were already broken.** The anomaly detectors' thresholds flag most real
  users, and the text models' thresholds, tuned while `#trending` was present, catch few fakes even
  with no attack.

## Account-stat models

**Common pattern:** the supervised models drop from about 0.82–0.85 to about 0.52 under the strong
combined attack. Once fakes copy real users' stats, the stats alone cannot separate them. Using
aged accounts, the one expensive attack, barely matters to any of them.

| Model | Result |
|---|---|
| **LightGBM** | Collapses (0.85 → 0.53). "Post less often" alone takes it to 0.70. Its explanations barely change while it fails. |
| **XGBoost** | Same pattern as LightGBM (0.85 → 0.52). Posting rate is its weak spot. |
| **Random forest** | Same pattern (0.84 → 0.52). Posting rate is its weak spot. |
| **Linear SVM** | Collapses (0.82 → 0.53). "Reply more" alone takes it to 0.70, the most of any single change. |
| **RBF SVM** | Collapses (0.85 → 0.53). Posting rate hurts most. |
| **NMF + logistic regression** | Collapses (0.77 → 0.53). "Reply more" alone takes it to 0.64; it barely reacts to posting rate. |
| **PCA + logistic regression** | Collapses (0.76 → 0.53), but no single change does much on its own: an attacker has to change several things. |
| **Isolation forest** | Weak to start with (0.63). A *mild* attack fools it more than a strong one (0.41 vs 0.52): half-disguised fakes look more typical than real users, and it flags the unusual. Its threshold flags 83% of real users. |
| **Local outlier factor** | Same as isolation forest: weak start (0.61), worse under mild (0.44) than strong (0.53), threshold flags most real users. |
| **One-class SVM** | Barely better than guessing before any attack (0.58); worse than guessing under mild (0.40). Threshold flags most real users. |
| **Dense autoencoder** | Barely better than guessing before any attack (0.56); worse than guessing under mild (0.44). Threshold flags nearly everyone. |

## Hourly-activity models

**Common pattern:** these models **only respond to volume**. Moving bursts, spreading them out or
faking a daily rhythm changes their AUC by about zero, while posting as much as a real user breaks
them. This confirms the "ignores timing" weakness found in the normal results.

| Model | Result |
|---|---|
| **1D CNN** | The most resistant activity model, but it still collapses (1.00 → 0.64). Matching volume alone takes it to 0.77. |
| **LSTM** | Collapses below chance (1.00 → 0.47). Matching volume alone takes it to 0.49. |
| **RNN** | Collapses below chance (0.89 → 0.48). Matching volume alone takes it to 0.45. |
| **LSTM autoencoder** | Collapses to chance (0.99 → 0.50). Matching volume alone takes it to 0.53. |

Under the strong combined attack these models catch 6–29% of fakes, against 94–99% with no attack.

## Text models

**Common pattern:** with `#trending` gone, every text model does *better* on fresh posts than on
the original test set, because the hashtag was misleading them. But changing the giveaway words
defeats all three. Disguises and synonyms have identical effects, since to these word-level
models both are just unknown words. All three decision thresholds are out of date: tuned while
`#trending` was present, they catch few fakes even with no attack.

| Model | Result |
|---|---|
| **Transformer** | Best with no attack (0.95, up from 0.87 on the original test set). Disguising the giveaway words alone drops it to a coin flip (0.51). Padding barely hurts it, because it reads only the first 14 words of a post. |
| **Doc2vec + logistic regression** | Strong with no attack (0.91), but the combined attack **inverts** it (0.04): it now rates disguised fakes as more genuine than real users. Synonyms alone take it to 0.46, and padding alone to 0.50. |
| **Word2vec + logistic regression** | Weakest text model (0.80 with no attack). Disguising the giveaway words takes it below chance (0.42); the combined attack takes it to 0.41. |

In all three, the credit given to the giveaway word falls from 40–80% of the total to about 0–10%
under the strong attack: the explanations show the signal is gone.

## Not on the website

The 7 graph models (GCN, GAT, GraphSAGE, R-GCN, node2vec, DeepWalk, label spreading) and the
multimodal fusion model were never attacked: attacking them would need new accounts wired into the
follower graph, which was left out of scope. They are therefore not shown on the website. Their
original fitting and explainability results remain in `outputs/step1_fit/` and
`outputs/step2_explain/`.
