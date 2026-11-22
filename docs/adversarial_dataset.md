# The adversarial dataset: how it was built

*Part 1 of 3. See also [the attacks](adversarial_attacks.md) and [the results by model](adversarial_results.md).*

## Why it exists

Every model in this project was trained and tested on synthetic accounts whose bots and trolls
behave "naturally": they post a lot, burst at fixed times, and use tell-tale words. Real operators
adapt. Once they learn what a detector looks for, they change exactly those behaviours. The
adversarial dataset asks the question a developer should ask before deploying: **what happens to
this model, and to its explanations, when the fakes start hiding?**

## Design principles

1. **Only the fakes change.** Bots and trolls are disguised; real users are left byte-for-byte
   identical in every version of the data. An attacked bot is still labelled a bot.
2. **A fresh population.** The test accounts are newly generated, never seen by any model. This
   separates "the model meets new accounts" from "the model meets disguised accounts".
3. **The same rules as the original data.** The population comes from the same generator
   (`anonsynth`) and is written in exactly the same file format, so every model reads it unchanged.
4. **No shortcut.** The `#trending` hashtag, which only signalled fakes in the original training
   data, is removed from every post.
5. **Known strength.** Every attack comes at two strengths, *mild* and *strong*, next to an
   unattacked baseline (*none*), so the damage can be read as a dose-response.
6. **Reproducible.** Everything is generated from a single seed, and tests check it.

## How it was generated

### Step 1: a new population

The generator creates 5,000 accounts with seed 1 (the original data uses seed 0), using the
original rules: account stats drawn from the same distributions, labels from the same formula
(about 30% fake), bots and trolls split as before, and the same posts, hourly activity, follower
graph and writing-style features. The one change is that the probability of `#trending` appearing
in any post is set to zero.

The accounts are split into training, validation and test sets in the same proportions as the
original data. **All results use the 991 test accounts (296 of them fake).** The training split
only serves as the pool of real users the attackers imitate.

This unattacked population is the *none* condition. Because it is new, the "no attack" numbers on
the website show how each model does on accounts it has never seen. For the account-stat and
activity models they match the original test results closely. For the text models they are
higher, because `#trending` no longer misleads them.

### Step 2: the attacks

Each attack is applied to every bot and troll in the population, at mild strength (0.5) and strong
strength (1.0). There are three families of attack, one per kind of input:

| Family | What is attacked | Attacks |
|---|---|---|
| Account stats | the 12 numbers describing each account | 6 single changes + 1 combined |
| Hourly activity | the 168 hourly post counts of each account's week | 4 single changes + 1 combined |
| Post text | the words of each account's posts | 3 single changes + 1 combined |

Each family only touches its own input. An activity attack never changes account stats or posts,
for example, so each model family is tested on exactly the input it reads.
[The attacks document](adversarial_attacks.md) describes every attack.

That gives 32 attacked versions of the population: 16 attacks, each at two strengths.

### Step 3: the answer key

For every attacked version the generator also records what it changed:

- **Account stats:** which real user each fake copied, and the original value of every changed stat.
- **Hourly activity:** the new position of the planted pattern (bursts move when they are shifted
  or spread), so explanations can still be scored against it.
- **Text:** each post's original text, which words were replaced and by what, and how many words
  were added.

## How the models are tested on it

- **Models are used exactly as trained.** The same saved files, checked to reproduce their original
  predictions before anything else runs. Inputs are scaled with the *original* training data's
  statistics, as a deployed model would be; nothing is re-fitted on the attacked data.
- **Accuracy** is measured on the test accounts for every attack and strength: AUC with a 95%
  interval, the share of fakes the model catches at its own decision threshold, and the share of
  real users it wrongly flags.
- **Explanations** are recomputed under the combined attack at all three strengths, with each
  model's own two methods and the same settings as the original explainability run. For the
  account-stat models that is SHAP and LIME on 50 humans, 50 bots and 50 trolls. For the activity
  models it is their gradient-based methods on the same kind of sample. For the text models it is
  SHAP and LIME on 60 posts by fake accounts.
- **Fixed rules** turn the results into the strengths and vulnerabilities shown on the site, as
  for the normal results ([docs/site_rules.md](site_rules.md), section 4).

## Checks

Automated tests (`tests/test_adversarial.py`, 18 tests) confirm that:

- the output has the same format as the original data, and the same seed gives the same output;
- real users and labels never change, and real users are statistically like the original
  population's real users;
- every attack changes only what it should, and in the right direction (halfway at mild, fully at
  strong where that applies);
- `#trending` appears nowhere, and no replacement word appears in the original posts;
- nothing leaks between the training, validation and test splits.

## Limits

- The data is synthetic. The attacks follow the rules of this generator, and real platforms are
  messier.
- The attacker is **black-box**: it imitates what real users visibly do, without seeing the
  models. An attacker who could query the models directly would do at least as well.
- Each family is attacked separately. A real operator would attack every input at once, which is
  what the planned graph and multimodal attacks would test.
- The follower graph was not attacked, so the graph and multimodal models have no adversarial
  results and are not shown on the website.

## Where things are

| What | Where |
|---|---|
| Generator code and technical notes | [`adversarial/`](../adversarial/README.md) |
| Generated data (not committed; rebuilt from the seed in about 10 seconds) | `adversarial/data/` after `python -m adversarial` |
| Results | `outputs/step3_adversarial/` |
| Rules for strengths and vulnerabilities | [`docs/site_rules.md`](site_rules.md) |
