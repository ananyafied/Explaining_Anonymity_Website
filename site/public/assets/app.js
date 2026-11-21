import { el, pct, num, legend, tableView, hbars, signedTable, hourChart, segBars, segTabs, hideTip } from "./charts.js";

const S1 = "var(--s1)", S2 = "var(--s2)", S3 = "var(--s3)", TRUTH = "var(--truth)";
const KIND_LABEL = { human: "Human", bot: "Bot", troll: "Troll" };
const SCORE_LABEL = {
  classifier: "probability of being fake",
  anomaly: "anomaly score (higher = more unusual)",
  propagation: "share of 'fake' label reaching the account",
};
const KIND_TEXT = {
  classifier: "Trained with labels to output the probability that an account is fake.",
  anomaly: "Trained without labels to score how unusual an account is.",
  propagation: "Spreads a few known labels along the graph to the other accounts.",
};
const ROLE_TAG = {
  informative_linear: "drives label", nonlinear_threshold: "threshold",
  distractor: "decoy", noise: "noise",
};
const roleTag = (role) => role.startsWith("interaction") ? "interaction" : role.startsWith("redundant_proxy") ? "near-copy" : ROLE_TAG[role];
const ATTACK_PHASE = {
  tabular: "Attack results are being generated.",
  sequences: "Coming soon: these models will face bots and trolls that post at human volume and rhythm.",
  text: "Coming soon: these models will face posts with disguised marker words and no #trending.",
  graph: "Planned (optional phase): graph attacks need new accounts wired into the network.",
  multimodal: "Planned (optional phase): this model will face the combined attacks.",
};

let CATALOG;
const $ = (s) => document.querySelector(s);
const normalise = (t) => t.trim().toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");

// ------------------------------------------------------------------ model picker (combobox)
function namesOf(m) { return [normalise(m.key), m.name.toLowerCase(), ...m.aliases]; }
function lookup(text) {
  const t = normalise(text);
  return CATALOG.models.find((m) => namesOf(m).includes(t)) || null;
}
function matches(text) {
  const t = normalise(text);
  if (!t) return CATALOG.models;
  return CATALOG.models.filter((m) => namesOf(m).some((n) => n.includes(t)));
}

function setupPicker() {
  const input = $("#model-input"), list = $("#model-list"), msg = $("#picker-msg");
  let active = -1, shown = [], navigated = false;

  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; };
  const choose = (m) => { close(); msg.textContent = ""; input.value = m.name; location.hash = `model=${m.key}`; };
  const highlight = (i) => {
    const opts = [...list.querySelectorAll(".opt")];
    opts.forEach((o, j) => o.setAttribute("aria-selected", String(j === i)));
    active = i;
    if (opts[i]) { input.setAttribute("aria-activedescendant", opts[i].id); opts[i].scrollIntoView({ block: "nearest" }); }
  };
  const open = () => {
    shown = matches(input.value);
    list.replaceChildren();
    if (!shown.length) list.append(el("li", { class: "empty", text: "No model matches. Try 'forest', 'svm', 'lstm' or 'gcn'." }));
    for (const fam of CATALOG.families) {
      const ms = shown.filter((m) => m.family === fam.key);
      if (!ms.length) continue;
      list.append(el("li", { class: "group", role: "presentation", text: fam.name }));
      for (const m of ms) {
        const i = shown.indexOf(m);
        list.append(el("li", { class: "opt", role: "option", id: `opt-${m.key}`, "aria-selected": "false",
          onmousedown: (e) => { e.preventDefault(); choose(m); }, onmousemove: () => highlight(i) },
          el("span", { text: m.name }), el("small", {}, el("span", { class: `pill ${m.verdict}`, text: m.verdict }))));
      }
    }
    // keep option order = visual order for keyboard navigation
    shown = [...list.querySelectorAll(".opt")].map((o) => CATALOG.models.find((m) => `opt-${m.key}` === o.id));
    list.hidden = false; input.setAttribute("aria-expanded", "true");
    highlight(shown.length && normalise(input.value) ? 0 : -1);
  };

  input.addEventListener("input", () => { msg.textContent = ""; navigated = false; open(); });
  input.addEventListener("focus", open);
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); if (list.hidden) open(); navigated = true; highlight(Math.min(active + 1, shown.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); navigated = true; highlight(Math.max(active - 1, 0)); }
    else if (e.key === "Escape") { close(); }
    else if (e.key === "Enter") {
      e.preventDefault();
      // an option picked with the arrow keys wins; otherwise an exact name or alias; otherwise the top match
      const m = (navigated && shown[active]) || lookup(input.value) || shown[active] || (shown.length === 1 ? shown[0] : null);
      if (m) choose(m);
      else msg.textContent = input.value.trim() ? `No model called "${input.value.trim()}". Pick one from the list.` : "";
    }
  });
  $("#model-toggle").addEventListener("click", () => { input.focus(); if (list.hidden) open(); });
  $("#examples").addEventListener("click", (e) => {
    const k = e.target.dataset && e.target.dataset.key;
    if (k) choose(CATALOG.models.find((m) => m.key === k));
  });
}

// ------------------------------------------------------------------ shared section builders
function section(id, title, sub, ...kids) {
  return el("section", { class: "block", id }, el("h2", { text: title }), sub ? el("p", { class: "sub", text: sub }) : null, ...kids);
}
const card = (...kids) => el("div", { class: "card" }, ...kids);
const tile = (label, value, detail) => el("div", { class: "tile" }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), detail ? el("div", { class: "detail", text: detail }) : null);

function headSection(m) {
  const p = m.performance;
  const gap = p.val_auc - p.test_auc;
  const checkName = { accuracy: "Accuracy", generalises: "Holds up on new data", right_reasons: "Right reasons" };
  return el("section", { class: "block", id: "summary" },
    card(
      el("div", { class: "model-head" }, el("h2", { text: m.name }), el("span", { class: `pill ${m.verdict.label}`, text: m.verdict.label }), el("span", { class: "tag", text: m.family_name })),
      el("p", { class: "meta", text: `Reads: ${m.input}. ${KIND_TEXT[m.kind]}` }),
      el("div", { class: "tiles" },
        tile("Test AUC", num(p.test_auc, 3), `95% interval ${num(p.test_auc_ci[0], 3)}–${num(p.test_auc_ci[1], 3)}`),
        tile("Reference AUC", num(p.reference_auc, 3), p.reference_how),
        tile("Validation → test", `${gap > 0 ? "−" : "+"}${num(Math.abs(gap), 3)}`, `validation ${num(p.val_auc, 3)}, test ${num(p.test_auc, 3)}`)),
      el("ul", { class: "checks", "aria-label": "Verdict checks" }, m.verdict.checks.map((c) =>
        el("li", { class: c.passed ? "pass" : "fail" }, el("span", { class: "mark", "aria-hidden": "true", text: c.passed ? "✓" : "✗" }),
          el("span", {}, el("b", { text: `${checkName[c.name]}${c.passed ? "" : " (not met)"}: ` }), c.detail)))),
      el("p", { class: "note", text: "AUC is the chance that a random fake account scores higher than a random real one: 0.5 is guessing, 1.0 is perfect. A model is labelled good only if it passes all three checks." })));
}

function methodsBlock(m) {
  const colours = [S1, S2];
  const info = el("div", { class: "methods" }, m.methods.map((name, i) =>
    el("div", { class: "method", "--mc": colours[i] }, el("b", { text: name }), el("p", { text: CATALOG.methods_info[name] }))));
  const a = m.scorecard.agreement.agreement_spearman;
  let agree = null;
  if (a != null) {
    const T = CATALOG.thresholds;
    const word = m.key === "label_spreading" ? "Agree by construction" : a >= T.agree_high ? "Agree" : a >= T.agree_low ? "Partly agree" : "Disagree";
    const cls = m.key === "label_spreading" ? "neutral" : a >= T.agree_high ? "good" : a >= T.agree_low ? "neutral" : "weak";
    agree = el("div", { class: "agree" }, el("span", { class: `pill ${cls}`, text: word }),
      el("span", { text: `The two methods' rankings of what matters have a rank correlation of ${num(a)} (1 = identical order, 0 = unrelated).` }));
  }
  return card(el("h3", { text: "How this model is explained" }), info, agree);
}

function scorecardTable(m) {
  const pm = m.scorecard.per_method;
  const crit = [...new Set(Object.values(pm).flatMap((v) => Object.keys(v)))].filter((c) => c !== "seconds_per_explanation");
  const rows = crit.map((c) => [CATALOG.criteria[c] || c, ...m.methods.map((meth) => num(pm[meth]?.[c], 3))]);
  for (const [c, v] of Object.entries(m.scorecard.agreement)) rows.push([`Agreement between methods: ${c.replace("agreement_", "")}`, num(v, 3), ""]);
  return tableView(["Measure (higher is better)", ...m.methods], rows, "Full explanation scorecard");
}

function findingsSection(m) {
  const list = (items, cls, mark) => items.length
    ? el("ul", { class: "findings" }, items.map((x) => el("li", {}, el("span", { class: "mark", "aria-hidden": "true", text: mark }), el("span", { text: x.text }))))
    : el("p", { class: "note", text: "None found." });
  return section("findings", "Strengths and weaknesses", "Generated from fixed rules applied to the numbers above; nothing is written by hand.",
    el("div", { class: "two-col" },
      el("div", { class: "card strengths" }, el("h3", { text: `Strengths (${m.strengths.length})` }), list(m.strengths, "s", "+")),
      el("div", { class: "card weaknesses" }, el("h3", { text: `Weaknesses (${m.weaknesses.length})` }), list(m.weaknesses, "w", "–"))));
}

function checklistSection(m) {
  const general = m.checklist.filter((c) => !c.because), specific = m.checklist.filter((c) => c.because);
  return section("checklist", "Before you deploy it", "What to test before putting this model in front of real users.",
    card(
      el("h3", { text: "For any account-detection model" }),
      el("ol", { class: "checklist" }, general.map((c) => el("li", { text: c.text }))),
      specific.length ? el("div", { class: "specific" },
        el("h3", { text: `Specific to ${m.name}` }),
        el("ol", { class: "checklist", start: general.length + 1 }, specific.map((c) => el("li", {}, c.text, el("span", { class: "because", text: `Because: ${c.because}` }))))) : null));
}

function attackSection(m) {
  const A = m.attack;
  if (!A) return section("attack", "Under attack", null, card(el("p", { class: "coming", style: { margin: 0 }, text: ATTACK_PHASE[m.family] })));
  const LEVELS = ["none", "mild", "strong"], LABEL = { none: "None", mild: "Mild", strong: "Strong" };
  const base = A.results.none, comb = A.results[A.combined];
  const at = (res, lvl) => (lvl === "none" ? base : res[lvl]);
  const delta = (x, y, f) => { const d = y - x; return `${d >= 0 ? "+" : "−"}${f(Math.abs(d))} vs no attack`; };
  const pctPts = (x) => `${(100 * x).toFixed(0)} pts`;

  // --- slider
  const slider = el("input", { type: "range", min: 0, max: 2, step: 1, value: 2, id: "attack-strength", "aria-label": "Attack strength", class: "strength" });
  const sliderLabels = el("div", { class: "strength-labels", "aria-hidden": "true" }, LEVELS.map((l) => el("span", { text: LABEL[l] })));
  const tiles = el("div", { class: "tiles" });
  const impBox = el("div"), singleBox = el("div");

  const feats = Object.keys(A.importance.SHAP.none).sort((a, b) => A.importance.SHAP.none[b] - A.importance.SHAP.none[a]);
  const singles = Object.keys(A.attacks).filter((k) => k !== A.combined);

  const render = () => {
    const lvl = LEVELS[Number(slider.value)];
    slider.setAttribute("aria-valuetext", LABEL[lvl]);
    const r = at(comb, lvl);
    tiles.replaceChildren(
      tile("AUC", num(r.auc, 3), lvl === "none" ? `95% interval ${num(r.auc_ci[0], 3)}–${num(r.auc_ci[1], 3)}` : delta(base.auc, r.auc, (x) => num(x, 3))),
      tile("Fakes caught", pct(r.caught), lvl === "none" ? "at the model's own decision threshold" : delta(base.caught, r.caught, pctPts)),
      tile("Real users flagged", pct(r.flagged_real), "real users never change, so neither does this"));

    const rows = feats.map((f) => ({ label: f, values: { SHAP: A.importance.SHAP[lvl][f], LIME: A.importance.LIME[lvl][f] }, truth: A.importance.SHAP.none[f] }));
    impBox.replaceChildren(
      legend([{ label: "SHAP", color: S1 }, { label: "LIME", color: S2 }, { label: "SHAP with no attack", kind: "tick" }]),
      hbars({ rows, series: [{ key: "SHAP", name: `SHAP (${LABEL[lvl].toLowerCase()})`, color: S1 }, { key: "LIME", name: `LIME (${LABEL[lvl].toLowerCase()})`, color: S2 }], truthName: "SHAP, no attack" }),
      tableView(["Feature", `SHAP (${LABEL[lvl]})`, `LIME (${LABEL[lvl]})`, "SHAP (None)"], rows.map((x) => [x.label, pct(x.values.SHAP, 1), pct(x.values.LIME, 1), pct(x.truth, 1)])));

    if (lvl === "none") {
      // with no attack every bar would equal the no-attack AUC, so there is nothing to compare
      singleBox.replaceChildren(el("p", { class: "note", text: "Move the slider to Mild or Strong to compare attacks." }));
      return;
    }
    const srows = [...singles, A.combined].map((k) => ({ label: A.attacks[k].name, tags: [A.attacks[k].cost], values: { auc: A.results[k][lvl].auc }, truth: base.auc }));
    singleBox.replaceChildren(
      el("p", { class: "note", text: `Showing ${LABEL[lvl].toLowerCase()} attacks.` }),
      legend([{ label: "AUC under the attack", color: S1 }, { label: "AUC with no attack", kind: "tick" }]),
      hbars({ rows: srows, series: [{ key: "auc", name: "AUC under attack", color: S1 }], truthName: "No attack", fmt: (x) => num(x, 3), max: 1 }),
      tableView(["Attack", "Cost", "Changes", "AUC (mild)", "AUC (strong)", "Fakes caught (strong)"],
        [...singles, A.combined].map((k) => [A.attacks[k].name, A.attacks[k].cost, A.attacks[k].changes.join(", "), num(A.results[k].mild.auc, 3), num(A.results[k].strong.auc, 3), pct(A.results[k].strong.caught)])));
  };
  slider.addEventListener("input", render);
  render();

  const list = (items, mark) => items.length
    ? el("ul", { class: "findings" }, items.map((x) => el("li", {}, el("span", { class: "mark", "aria-hidden": "true", text: mark }), el("span", { text: x.text }))))
    : el("p", { class: "note", text: "None found." });

  return section("attack", "Under attack",
    `Bots and trolls copy real users' habits (mimicry). Tested on a fresh population the model has never seen (${A.n} test accounts, ${A.n_fake} of them fake), with #trending removed everywhere. Real users are never changed.`,
    card(el("h3", { text: `${A.attacks[A.combined].name}` }),
      el("p", { class: "note", style: { marginTop: 0 }, text: `${A.attacks[A.combined].how} Changes: ${A.attacks[A.combined].changes.join(", ")}. Mild moves each feature halfway to a real user's value; strong copies it.` }),
      el("div", { class: "strength-wrap" }, el("label", { for: "attack-strength", text: "Attack strength" }), slider, sliderLabels),
      tiles),
    card(el("h3", { text: "What it relies on under this attack" }),
      el("p", { class: "note", style: { marginTop: 0 }, text: "Share of credit per feature for 50 humans, 50 bots and 50 trolls from the attacked population, at the strength chosen above." }), impBox),
    card(el("h3", { text: "Which change hurts it most?" }), singleBox),
    el("div", { class: "two-col", style: { marginTop: "12px" } },
      el("div", { class: "card strengths" }, el("h3", { text: `Holds up (${A.strengths.length})` }), list(A.strengths, "+")),
      el("div", { class: "card weaknesses" }, el("h3", { text: `Vulnerabilities (${A.weaknesses.length})` }), list(A.weaknesses, "–"))),
    A.checklist.length ? card(el("h3", { text: "Attack tests to run before deploying" }),
      el("ol", { class: "checklist" }, A.checklist.map((c) => el("li", {}, c.text, el("span", { class: "because", text: `Because: ${c.because}` }))))) : null);
}

// ------------------------------------------------------------------ family views
function tabularView(m) {
  const v = m.view, [M1, M2] = m.methods;
  const feats = [...v.features].sort((a, b) => (v.true_importance[b.name] - v.true_importance[a.name]) || (v.importance[M1][b.name] - v.importance[M1][a.name]));
  const rows = feats.map((f) => ({ label: f.name, tags: [roleTag(f.role)], values: { [M1]: v.importance[M1][f.name], [M2]: v.importance[M2][f.name] }, truth: v.true_importance[f.name] }));
  const series = [{ key: M1, name: M1, color: S1 }, { key: M2, name: M2, color: S2 }];
  const credit = m.evidence.credit;
  const groupLine = (meth) => el("li", { text: `${meth}: ${pct(credit[meth].informative)} on the 6 that drive the label · ${pct(credit[meth].proxy)} near-copy · ${pct(credit[meth].decoys)} decoys · ${pct(credit[meth].noise)} noise` });

  const importance = card(
    el("h3", { text: "Share of credit per feature" }),
    legend([{ label: M1, color: S1 }, { label: M2, color: S2 }, { label: "True share", kind: "tick" }]),
    hbars({ rows, series }),
    el("ul", { class: "note", style: { paddingLeft: "18px" } }, groupLine(M1), groupLine(M2)),
    el("p", { class: "note", text: `Credit is each feature's average absolute attribution over ${v.n_explained} test accounts (50 humans, 50 bots, 50 trolls), as a share of the total. The true share comes from the formula that generated the labels (the interaction is counted for both of its inputs): ${v.formula}` }),
    tableView(["Feature", "Role", M1, M2, "True share"], rows.map((r) => [r.label, r.tags[0], pct(r.values[M1], 1), pct(r.values[M2], 1), pct(r.truth, 1)])));

  const exBox = el("div");
  const renderEx = (kind) => {
    const ex = v.examples.find((e) => e.kind === kind);
    const exRows = feats.map((f) => ({ label: f.name, tags: [roleTag(f.role)], note: num(ex.values[f.name], 3),
      cells: { [M1]: ex.attributions[M1][f.name], [M2]: ex.attributions[M2][f.name], truth: ex.true_contribution[f.name] } }));
    exBox.replaceChildren(
      el("p", { class: "meta", text: `Account ${ex.account_id} · model score ${num(ex.score, 3)} (${SCORE_LABEL[m.kind]})` }),
      signedTable({ rows: exRows, cols: [{ key: M1, name: M1, color: S1 }, { key: M2, name: M2, color: S2 }, { key: "truth", name: "True contribution", color: TRUTH }], noteHeader: "Value" }),
      el("p", { class: "note", text: "Each column is scaled to its own largest value. Bars to the right push toward 'fake'. True contributions are in the formula's own units and are zero for features with no effect." }));
  };
  const examples = card(el("h3", { text: "One account, explained" }),
    segTabs(v.examples.map((e) => ({ key: e.kind, label: KIND_LABEL[e.kind] })), "bot", renderEx), exBox);
  renderEx("bot");
  return [importance, examples];
}

function sequencesView(m) {
  const v = m.view, [M1, M2] = m.methods;
  const inside = v.attribution_inside_planted;
  const stats = card(el("h3", { text: "Does the credit land on the planted bursts?" }),
    el("div", { class: "tiles" },
      tile("Planted hours", pct(v.planted_hour_share), "share of a bot's or troll's week"),
      tile(`${M1}: credit inside`, pct(inside[M1]), "share of its credit on planted hours"),
      tile(`${M2}: credit inside`, pct(inside[M2]), "share of its credit on planted hours")),
    el("p", { class: "note", text: `Bots burst every ${v.planted_pattern.bot_periods.join(", ")} hours; each troll ring shares two ${v.planted_pattern.troll_window_len}-hour windows. A method that finds the pattern puts far more than ${pct(v.planted_hour_share)} of its credit there.` }));

  const box = el("div");
  const render = (kind) => {
    const ex = v.examples.find((e) => e.kind === kind);
    const series = [{ name: M1, color: S1, values: ex.attributions[M1] }, { name: M2, color: S2, values: ex.attributions[M2] }];
    box.replaceChildren(
      el("p", { class: "meta", text: `Account ${ex.account_id} · model score ${num(ex.score, 3)} (${SCORE_LABEL[m.kind]})${kind === "human" ? " · humans have no planted pattern" : ""}` }),
      legend([{ label: "Posts", color: "var(--muted)" }, { label: "Planted hours", kind: "wash" }, { label: M1, color: S1, kind: "line" }, { label: M2, color: S2, kind: "line" }]),
      hourChart({ counts: ex.counts, planted: ex.planted, series }),
      tableView(["Hour", "Posts", "Planted", M1, M2], ex.counts.map((c, h) => [`Day ${Math.floor(h / 24) + 1} ${String(h % 24).padStart(2, "0")}:00`, String(c), ex.planted[h] ? "yes" : "", num(Math.abs(ex.attributions[M1][h]), 2), num(Math.abs(ex.attributions[M2][h]), 2)])));
  };
  const examples = card(el("h3", { text: "One week, explained" }),
    el("p", { class: "note", text: "Each method's credit is scaled to its own peak for this account. Hover or use the arrow keys for exact values." }),
    segTabs(v.examples.map((e) => ({ key: e.kind, label: KIND_LABEL[e.kind] })), "bot", render), box);
  render("bot");
  return [stats, examples];
}

function textView(m) {
  const v = m.view, [M1, M2] = m.methods;
  const rel = m.evidence.reliance;
  const reliance = card(el("h3", { text: "What the model really uses" }),
    el("div", { class: "tiles" },
      tile("AUC per post", num(rel.auc_original, 3), "all test posts, as they are"),
      tile("Without #trending", num(rel.auc_without_decoy, 3), rel.drop_without_decoy < 0 ? "rises: the hashtag was misleading it" : "falls: it relied on the hashtag"),
      tile("Without marker words", num(rel.auc_without_markers, 3), "falls toward 0.5 if it relies on them")),
    el("p", { class: "note", text: "Single posts are harder than whole accounts, so these AUCs are lower than the account-level AUC above. In the test data #trending appears equally in real and fake posts." }));

  const tagsFor = (w) => [w.is_marker ? "marker" : null, w.is_trending ? "trap" : null].filter(Boolean);
  const topCards = el("div", { class: "two-col" }, [M1, M2].map((meth, i) => {
    const words = v.top_words[meth];
    return el("div", {}, el("h3", { style: { fontSize: "15px" }, text: meth }),
      hbars({ rows: words.map((w) => ({ label: w.word, tags: tagsFor(w), values: { [meth]: w.mean_abs } })), series: [{ key: meth, name: `${meth} (mean |credit|)`, color: i ? S2 : S1 }], fmt: (x) => num(x, 2) }));
  }));
  const top = card(el("h3", { text: "Words that get the most credit" }), topCards,
    el("p", { class: "note", text: `Average absolute credit over the explained test posts (words seen at least 3 times). Planted marker words: ${v.markers.join(", ")}. The trap is ${v.trending_token}, which only signalled fake accounts in the training data.` }));

  const STRATUM = { marker: "A post with a marker word", spurious_only: `A post whose only signal is ${v.trending_token}`, neutral: "An ordinary post" };
  const exCards = v.examples.map((ex) => {
    const rows = ex.words.map((w, j) => ({ label: w, tags: [v.markers.includes(w) ? "marker" : null, w === v.trending_token ? "trap" : null].filter(Boolean),
      cells: { [M1]: ex.attributions[M1]?.[j], [M2]: ex.attributions[M2]?.[j] } }));
    return el("div", { style: { marginTop: "14px" } }, el("h3", { style: { fontSize: "15px" }, text: STRATUM[ex.stratum] || ex.stratum }),
      signedTable({ rows, cols: [{ key: M1, name: M1, color: S1 }, { key: M2, name: M2, color: S2 }], firstHeader: "Word" }));
  });
  const examples = card(el("h3", { text: "Single posts, explained" }), ...exCards,
    el("p", { class: "note", text: "Each column is scaled to its own largest value. Bars to the right push toward 'fake'." }));
  return [reliance, top, examples];
}

function graphView(m) {
  const ev = m.evidence, out = [];
  const rel = ev.reliance || {};
  if (m.view.troll_edge_importance) {
    const ti = m.view.troll_edge_importance, [M1, M2] = m.methods;
    const rows = [
      { label: "Edges inside the troll's ring", values: { [M1]: ti[M1].ring_edges, [M2]: ti[M2].ring_edges } },
      { label: "Other edges", values: { [M1]: ti[M1].other_edges, [M2]: ti[M2].other_edges } },
    ];
    out.push(card(el("h3", { text: "Do the explanations find troll rings?" }),
      legend([{ label: M1, color: S1 }, { label: M2, color: S2 }]),
      hbars({ rows, series: [{ key: M1, name: M1, color: S1 }, { key: M2, name: M2, color: S2 }], fmt: (x) => num(x, 2), max: 1 }),
      el("p", { class: "note", text: `Average importance of each edge around 50 explained trolls, scaled so each troll's most important edge is 1 (${ti[M1].n_ring} ring edges, ${ti[M1].n_other} others). Ring edges should score clearly higher.` }),
      tableView(["Edges", M1, M2], rows.map((r) => [r.label, num(r.values[M1], 3), num(r.values[M2], 3)]))));
  }
  if (ev.embedding_probe) {
    out.push(card(el("h3", { text: "What do the embedding dimensions encode?" }),
      el("div", { class: "tiles" },
        tile("Predicts degree", `R² ${num(ev.embedding_probe.r2_predicting_log_degree)}`, "how much of an account's connection count the 32 dimensions encode"),
        tile("Predicts ring membership", `AUC ${num(ev.embedding_probe.auc_predicting_troll_ring_membership)}`, "whether an account is in a troll ring")),
      el("p", { class: "note", text: "SHAP and LIME can only say which of 32 unnamed dimensions mattered. This probe is the closest the model gets to a human-readable explanation." })));
  }
  out.push(card(el("h3", { text: "How much does it lean on the graph, and on degree?" }),
    el("div", { class: "tiles" },
      tile("Score vs degree", num(rel.spearman_with_degree), "rank correlation with the number of connections"),
      tile("Degree alone", `AUC ${num(ev.degree_alone_auc, 3)}`, "counting connections, nothing else"),
      rel.auc_half_edges != null ? tile("Half the edges removed", `AUC ${num(rel.auc_half_edges, 3)}`, `vs ${num(m.performance.test_auc, 3)} with all edges`) : null),
    el("p", { class: "note", text: "Bots and trolls have more connections than humans in this data, so counting connections is a strong shortcut. A model that mostly counts them will flag popular real accounts too." })));
  return out;
}

function multimodalView(m) {
  const v = m.view, ev = m.evidence;
  const names = v.blocks, colours = { tab: S1, text: S2, graph: S3 };
  const cards = m.methods.map((meth) => {
    const rows = ["human", "bot", "troll"].map((k) => ({ who: KIND_LABEL[k], parts: ["tab", "text", "graph"].map((b) => ({ name: names[b], value: v.block_share_by_kind[meth][k][b], color: colours[b] })) }));
    return el("div", {}, el("h3", { style: { fontSize: "15px" }, text: meth }), segBars(rows));
  });
  const shares = card(el("h3", { text: "Which kind of evidence does it rely on?" }),
    legend(["tab", "text", "graph"].map((b) => ({ label: names[b], color: colours[b] }))), ...cards,
    el("p", { class: "note", text: "Share of credit per input block, averaged over test accounts of each kind. Trolls should lean on the graph (their rings); bots and trolls on text (their marker words)." }),
    tableView(["Method", "Kind", ...["tab", "text", "graph"].map((b) => names[b])], m.methods.flatMap((meth) => ["human", "bot", "troll"].map((k) => [meth, KIND_LABEL[k], ...["tab", "text", "graph"].map((b) => pct(v.block_share_by_kind[meth][k][b]))]))));
  const label = (key) => key === "(nothing)" ? "Nothing" : key.split("+").map((b) => names[b]).join(" + ");
  const co = Object.entries(ev.coalition_auc).filter(([k]) => k !== "(nothing)").sort((a, b) => b[1] - a[1]);
  const coal = card(el("h3", { text: "How well each combination of inputs works" }),
    hbars({ rows: co.map(([k, a]) => ({ label: label(k), values: { auc: a } })), series: [{ key: "auc", name: "Test AUC", color: S1 }], fmt: (x) => num(x, 3), max: 1 }),
    el("p", { class: "note", text: "The model's test AUC with only some input blocks switched on (the others replaced by their training average or an empty graph). Partly off-distribution, since the model always saw all three in training." }),
    tableView(["Inputs", "Test AUC"], co.map(([k, a]) => [label(k), num(a, 3)])));
  return [shares, coal];
}

const FAMILY_VIEW = { tabular: tabularView, sequences: sequencesView, text: textView, graph: graphView, multimodal: multimodalView };

// ------------------------------------------------------------------ pages
async function renderModel(key) {
  const meta = CATALOG.models.find((m) => m.key === key);
  const out = $("#content");
  if (!meta) { renderOverview(`No model called "${key}".`); return; }
  $("#model-input").value = meta.name;
  out.style.opacity = ".5";
  let m;
  try {
    const r = await fetch(`data/models/${key}.json`);
    if (!r.ok) throw new Error(r.status);
    m = await r.json();
  } catch {
    out.style.opacity = "1";
    out.replaceChildren(card(el("p", { text: "This model's data could not be loaded. Please try again." })));
    return;
  }
  m.input = meta.input;
  const fam = CATALOG.families.find((f) => f.key === m.family);
  out.replaceChildren(
    headSection(m),
    section("relies", "What it relies on", fam.about, methodsBlock(m), ...FAMILY_VIEW[m.family](m), card(scorecardTable(m))),
    findingsSection(m),
    checklistSection(m),
    attackSection(m));
  out.style.opacity = "1";
  document.title = `${m.name} · Explaining Anonymity`;
}

function renderOverview(message) {
  document.title = "Explaining Anonymity";
  $("#model-input").value = "";
  const cards = CATALOG.families.map((fam) => {
    const ms = CATALOG.models.filter((m) => m.family === fam.key);
    const good = ms.filter((m) => m.verdict === "good").length;
    return card(el("h3", {}, el("span", { text: fam.name }), el("span", { class: "tag", text: `${good} of ${ms.length} good` })),
      el("p", { class: "about", text: fam.about }),
      el("ul", {}, ms.map((m) => el("li", {}, el("a", { href: `#model=${m.key}`, text: m.name }),
        el("span", { style: { whiteSpace: "nowrap" } }, el("span", { class: "tag", style: { marginRight: "6px" }, text: `AUC ${num(m.test_auc, 2)}` }), el("span", { class: `pill ${m.verdict}`, text: m.verdict }))))));
  });
  $("#content").replaceChildren(
    ...(message ? [el("p", { class: "picker-msg", text: message })] : []),
    section("all", "All models", "Every model was trained on the same synthetic accounts, where we know exactly which signals make an account fake. That lets us check whether each explanation is right, not just whether it looks plausible.",
      el("div", { class: "overview" }, cards)));
}

function route() {
  hideTip();
  const key = new URLSearchParams(location.hash.slice(1)).get("model");
  if (key) renderModel(key); else renderOverview();
}

async function main() {
  try {
    CATALOG = await (await fetch("data/catalog.json")).json();
  } catch {
    $("#content").replaceChildren(card(el("p", { text: "The model list could not be loaded. Please refresh the page." })));
    return;
  }
  setupPicker();
  window.addEventListener("hashchange", () => { route(); window.scrollTo({ top: 0 }); });
  route();
}
main();
