# The attacks

*Part 2 of 3. See also [how the dataset was built](adversarial_dataset.md) and [the results by model](adversarial_results.md).*

Every attack disguises **bots and trolls only**; real users are never changed. Each one is run at
two strengths:

- **Mild** (strength 0.5): the change is applied halfway, or to about half of what it could touch.
- **Strong** (strength 1.0): the change is applied fully.

Each attack also has a **cost** for a real operator: *cheap* (a setting or a script), *moderate*
(more effort or lost effectiveness) or *expensive* (money or months of waiting). A model that a
cheap attack can defeat is more at risk in practice than one that only an expensive attack can.

Each family has a **combined** attack that applies its changes together. The website's strength
slider controls the combined attack.

---

## 1. Account-stat attacks

**Targets:** the 11 models that read the 12 account stats (LightGBM, XGBoost, random forest, the
two SVMs, NMF and PCA + logistic regression, and the four anomaly detectors).

**How they work: mimicry.** Each fake account picks one real user (its "donor") and moves the
attacked stats toward that person's values. Mild moves each stat halfway; strong copies it. The
copying respects each stat's natural range: counts and ages stay positive, fractions stay between
0 and 1. One donor per fake keeps the copied habits consistent with each other.

| Attack | What the fake changes | Stat changed | Cost |
|---|---|---|---|
| **Post less often** | Throttles its posting rate to a normal user's. Costs reach, not money. | `posts_per_day` (and `posts_per_week` with it) | cheap |
| **Reply more** | Replies to others as often as people do. Automated replies are easy. | `reply_fraction` | cheap |
| **Share fewer links** | Posts fewer links per post. | `url_fraction` | cheap |
| **Post in the daytime** | Schedules posts for hours when people are awake. | `night_activity_frac` | cheap |
| **Post steadily** | Spreads posts out instead of firing them in bursts. | `burstiness` | moderate |
| **Use aged accounts** | Buys old accounts, or waits months before using new ones. | `account_age_days` | expensive |
| **All cheap and moderate changes** (combined) | Everything above except aged accounts, which a typical operator avoids. | the five stats above | moderate |

The two decoy stats and three noise stats are not attacked. They are unrelated to being fake, so
fakes already look like real users on them.

---

## 2. Hourly-activity attacks

**Targets:** the 4 models that read a week of hourly post counts (1D CNN, LSTM, RNN, LSTM autoencoder).

**What they disguise.** In the original data, bots post a flat background with bursts every 6, 8 or
12 hours. Trolls post with a daily rhythm plus two 3-hour windows shared by their whole ring. Both
post far more than real users. These attacks work directly on the 168 hourly counts.

| Attack | What the fake changes | Details | Cost |
|---|---|---|---|
| **Move bursts to random times** | Breaks its fixed rhythm (bots) or stops posting in sync with its ring (trolls). | Each burst (a bot's burst hour, a troll's 3-hour window) moves to a random free spot. Mild: each burst moves with 50% chance. Strong: all move. Weekly total unchanged. | cheap |
| **Spread bursts out** | Makes no single hour unusually busy. | A share of each burst's extra posts moves to the two hours on either side: 37.5% at mild, 75% at strong. Weekly total unchanged. | cheap |
| **Fake a daily rhythm** | Makes background posting follow a human day, busier in the evening, instead of staying flat. | Bots only (trolls already have a rhythm). Each background post is re-timed with 50% (mild) or 100% (strong) chance. Weekly total unchanged. | cheap |
| **Post as much as a real user** | Cuts the week's volume to a normal person's. | The week is thinned out (or topped up), keeping its shape, so its total moves halfway (mild) or all the way (strong) to a real user's weekly total. | cheap |
| **All activity changes** (combined) | All four, in that order. | | cheap |

The first three change *when* posts happen; the fourth changes *how many* there are. That split
is what reveals whether a model reads timing patterns or just counts posts.

---

## 3. Text attacks

**Targets:** the 3 models that read posts (transformer, doc2vec + logistic regression, word2vec +
logistic regression).

**What they disguise.** In the original data, bots use words like *giveaway, promo, crypto, click,
winner, dm*, and trolls use words like *idiot, sheep, liar, shill, clown, wake*, each in about a
third of their posts. These "giveaway words" are the real text signal. (`#trending` has already
been removed from all posts.)

| Attack | What the fake changes | Details | Cost |
|---|---|---|---|
| **Disguise the giveaway words** | Misspells the tell-tale words. | `giveaway` → `g1veaway`, `promo` → `pr0mo`, `sheep` → `sh33p`, `idiot` → `1diot`, and so on. Each occurrence changed with 50% (mild) or 100% (strong) chance. | cheap |
| **Swap in synonyms** | Replaces the tell-tale words with ordinary words the models never saw in training. | `giveaway` → `freebie`, `crypto` → `bitcoin`, `idiot` → `fool`, `liar` → `fraud`, and so on. Same chances as above. | cheap |
| **Pad posts with ordinary words** | Adds everyday words so the tell-tale word is a smaller part of each post. | Ordinary words inserted at random places: +50% of the post's length at mild, +100% at strong. | cheap |
| **All text changes** (combined) | Disguises or swaps each giveaway word (half each) and pads every post. | | cheap |

Two things to know when reading the text results:

- **Disguises and synonyms look the same to these models.** All three read whole words, and both
  kinds of replacement are words they have never seen, so they treat both as an unknown word. A
  model that reads characters might catch a misspelling but not a synonym.
- **The transformer reads only the first 14 words of a post** (its training limit). A padded post
  is cut at 14 words, which sometimes removes the giveaway word entirely, just as a real length
  limit would.

---

## Summary

| Family | Single attacks | Combined attack | Models tested |
|---|---|---|---|
| Account stats | 6 (4 cheap, 1 moderate, 1 expensive) | 5 changes together | 11 |
| Hourly activity | 4 (all cheap) | all 4 together | 4 |
| Post text | 3 (all cheap) | all 3 together | 3 |

The graph and multimodal models were not attacked (graph attacks would need new accounts wired
into the network) and are not shown on the website.
