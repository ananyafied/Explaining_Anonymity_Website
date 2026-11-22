"""Build the Explaining Anonymity poster: poster/poster.html, then poster/poster.pdf via headless Chrome.

    python poster/build_poster.py

Every number on the poster is read from the saved results (outputs/ and site/public/data/), so a
rebuild after the results change stays correct. Edit the copy in TEXT below.
"""
from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SITE_URL = "https://ananyafied.github.io/Explaining_Anonymity_Website/"
REPO_URL = "github.com/ananyafied/Explaining_Anonymity_Website"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

W_IN, H_IN = 36, 24  # poster size in inches (landscape)
W, H = W_IN * 96, H_IN * 96  # CSS px at 96 px/in

# palette (validated reference slots, light mode) + ink
BLUE, ORANGE, AQUA, NOISE = "#2a78d6", "#eb6834", "#1baf7a", "#b9b6ad"
INK, INK2, MUTED, RULE, PAPER, PANEL = "#16181c", "#3d4047", "#62615c", "#d9d6cd", "#fbfaf7", "#f1efe9"
ACCENT = "#0f766e"

e = html.escape


def model_json(key):
    return json.loads((ROOT / "site" / "public" / "data" / "models" / f"{key}.json").read_text())


CATALOG = json.loads((ROOT / "site" / "public" / "data" / "catalog.json").read_text())
NAMES = {m["key"]: m["name"] for m in CATALOG["models"]}
FAMILY_OF = {m["key"]: m["family"] for m in CATALOG["models"]}


# ------------------------------------------------------------------ chart A: where SHAP puts the credit
def credit_chart():
    df = pd.read_csv(ROOT / "outputs" / "step2_explain" / "tabular" / "credit_by_feature_group.csv")
    df = df[(df["method"] == "SHAP") & df["model"].isin(NAMES)]
    df = df.rename(columns={"informative (6)": "inf", "proxy posts_per_week": "proxy", "decoys (2)": "dec", "noise (3)": "noise"})
    df = df.sort_values("inf", ascending=False)
    groups = [("inf", "Real signal (6 features)", BLUE, "#fff"), ("proxy", "Near-copy", ORANGE, "#fff"),
              ("dec", "Decoys", AQUA, "#0b0b0b"), ("noise", "Noise", NOISE, "#0b0b0b")]
    label_w, bar_w, row_h, bar_h = 380, 600, 48, 32
    rows = []
    for i, r in enumerate(df.itertuples()):
        y = i * row_h
        x = label_w
        segs = []
        for col, _, color, ink in groups:
            v = getattr(r, col)
            w = bar_w * v
            if w <= 0:
                continue
            segs.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 3, 0):.1f}" height="{bar_h}" rx="5" fill="{color}"/>')
            if col == "inf":
                segs.append(f'<text x="{x + 12:.1f}" y="{y + bar_h / 2 + 8}" font-size="23" font-weight="700" fill="{ink}">{v * 100:.0f}%</text>')
            x += w
        rows.append(f'<text x="{label_w - 16}" y="{y + bar_h / 2 + 8}" font-size="24" text-anchor="end" fill="{INK}">{e(NAMES[r.model])}</text>' + "".join(segs))
    h = len(df) * row_h
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{e(n)}</span>' for _, n, c, _ in groups)
    svg = f'<svg viewBox="0 0 {label_w + bar_w + 10} {h}" width="{label_w + bar_w + 10}" height="{h}" role="img" aria-label="Share of SHAP credit by feature group for 11 models">{"".join(rows)}</svg>'
    return f'<div class="legend">{legend}</div>{svg}'


# ------------------------------------------------------------------ chart B: AUC as the disguise gets stronger
def attack_chart():
    fams = [("tabular", "Account stats", "lightgbm"), ("sequences", "Hourly activity", "lstm"), ("text", "Post text", "transformer")]
    pw, ph, pad_l, pad_b, pad_t = 330, 360, 60, 56, 44
    xs = [pad_l + 34, pad_l + (pw - pad_l) / 2, pw - 44]
    yv = lambda v: pad_t + (ph - pad_t - pad_b) * (1 - v)  # noqa: E731  (AUC 0..1)
    panels = []
    for fam, title, hero in fams:
        keys = [k for k in NAMES if FAMILY_OF[k] == fam]
        parts = []
        for t in (0, 0.5, 1.0):
            parts.append(f'<line x1="{pad_l}" x2="{pw}" y1="{yv(t):.1f}" y2="{yv(t):.1f}" stroke="{RULE}" stroke-width="{2 if t == 0.5 else 1.5}"/>')
            parts.append(f'<text x="{pad_l - 10}" y="{yv(t) + 8:.1f}" font-size="22" text-anchor="end" fill="{MUTED}">{t:.1f}</text>')
        for k in sorted(keys, key=lambda k: k == hero):
            a = model_json(k)["attack"]
            c = a["results"][a["combined"]]
            vals = [a["results"]["none"]["auc"], c["mild"]["auc"], c["strong"]["auc"]]
            pts = " ".join(f"{x:.1f},{yv(v):.1f}" for x, v in zip(xs, vals))
            if k == hero:
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>')
                parts += [f'<circle cx="{x:.1f}" cy="{yv(v):.1f}" r="9" fill="{BLUE}" stroke="{PAPER}" stroke-width="3"/>' for x, v in zip(xs, vals)]
                parts.append(f'<text x="{xs[0] - 10}" y="{yv(vals[0]) - 22:.1f}" font-size="24" font-weight="700" fill="{BLUE}">{e(NAMES[k])}</text>')
            else:
                parts.append(f'<polyline points="{pts}" fill="none" stroke="#c4c1b8" stroke-width="3" stroke-linejoin="round"/>')
        for x, lab in zip(xs, ("None", "Mild", "Strong")):
            parts.append(f'<text x="{x:.1f}" y="{ph - 14}" font-size="22" text-anchor="middle" fill="{INK2}">{lab}</text>')
        n = len(keys)
        panels.append(f'<div class="panel"><div class="ptitle">{e(title)}</div><div class="pcount">{n} models</div>'
                      f'<svg viewBox="0 0 {pw} {ph}" width="{pw}" height="{ph}" role="img" aria-label="{e(title)}: AUC under no, mild and strong disguise">{"".join(parts)}</svg></div>')
    return f'<div class="panels">{"".join(panels)}</div>'


# ------------------------------------------------------------------ numbers quoted in the copy
def facts():
    lgb = model_json("lightgbm")
    shap_share = lgb["evidence"]["credit"]["SHAP"]["informative"]
    inter = lgb["evidence"]["interaction"]
    tr = model_json("transformer")
    tr_val, tr_test = tr["performance"]["val_auc"], tr["performance"]["test_auc"]
    ta = tr["attack"]["results"]
    lstm = model_json("lstm")["attack"]["results"]
    la = lgb["attack"]["results"]
    shifts = []
    for k in NAMES:
        if FAMILY_OF[k] != "tabular":
            continue
        for w in model_json(k)["attack"]["weaknesses"]:
            if w["id"] == "explanations_hide_attack":
                shifts.append(w["evidence"]["shift"])
    iso = model_json("isolation_forest")["attack"]["results"]
    timing = max(model_json("lstm")["evidence"]["timing"]["auc_drop"], 0)
    good = sum(1 for m in CATALOG["models"] if m["verdict"] == "good")
    return {
        "shap_share": f"{shap_share * 100:.0f}%",
        "inter_shap": f"{inter['SHAP'] * 100:.0f}%", "inter_true": f"{inter['true'] * 100:.0f}%",
        "tr_val": f"{tr_val:.3f}", "tr_test": f"{tr_test:.3f}",
        "timing": f"{timing:.3f}",
        "lgb_none": f"{la['none']['auc']:.2f}", "lgb_strong": f"{la['combined']['strong']['auc']:.2f}",
        "lgb_post": f"{la['posting_rate']['strong']['auc']:.2f}",
        "lgb_caught0": f"{la['none']['caught'] * 100:.0f}%", "lgb_caught1": f"{la['combined']['strong']['caught'] * 100:.0f}%",
        "lstm_none": f"{lstm['none']['auc']:.2f}", "lstm_vol": f"{lstm['match_volume']['strong']['auc']:.2f}",
        "lstm_shift": f"{lstm['shift_bursts']['strong']['auc']:.2f}",
        "tr_none": f"{ta['none']['auc']:.2f}", "tr_disg": f"{ta['disguise_markers']['strong']['auc']:.2f}",
        "shift_lo": f"{min(shifts) * 100:.0f}", "shift_hi": f"{max(shifts) * 100:.0f}",
        "iso_mild": f"{iso['combined']['mild']['auc']:.2f}", "iso_strong": f"{iso['combined']['strong']['auc']:.2f}",
        "good": good, "n_models": len(NAMES),
    }


def build_html():
    F = facts()
    screenshot = "../docs/img/site_model_page.png"  # relative to poster/poster.html, so no local path ends up in the file
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Explaining Anonymity Poster</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcode-generator/1.4.4/qrcode.min.js"></script>
<style>
@page {{ size: {W_IN}in {H_IN}in; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; background: {PAPER}; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ width: {W}px; height: {H}px; overflow: hidden; color: {INK}; font-family: Inter, system-ui, sans-serif; font-size: 30px; line-height: 1.4; }}
h1, h2, .big, .ptitle {{ font-family: "Space Grotesk", Inter, sans-serif; }}
.poster {{ width: {W}px; height: {H}px; display: grid; grid-template-rows: 370px minmax(0, 1fr) 160px; }}
header {{ background: {INK}; color: #fff; display: grid; grid-template-columns: 1fr auto; align-items: center; padding: 0 96px; gap: 64px; }}
header h1 {{ margin: 0; font-size: 104px; line-height: 1.02; letter-spacing: -0.02em; }}
header h1 .t2 {{ display: block; font-size: 66px; font-weight: 600; color: #e9e6de; margin-top: 8px; letter-spacing: -0.01em; }}
header .who {{ font-size: 32px; color: #b9b6ad; margin-top: 22px; }}
header .qr {{ display: flex; align-items: center; gap: 28px; }}
header .qr .box {{ background: #fff; padding: 16px; border-radius: 12px; width: 270px; height: 270px; }}
header .qr .box svg {{ width: 100%; height: 100%; display: block; }}
header .qr .txt {{ font-size: 26px; color: #d9d6cd; max-width: 330px; }}
header .qr .txt b {{ color: #fff; font-size: 32px; display: block; margin-bottom: 6px; }}
main {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 72px; padding: 52px 96px 40px; min-height: 0; }}
section {{ display: flex; flex-direction: column; gap: 26px; min-width: 0; }}
.num {{ display: inline-flex; width: 64px; height: 64px; border-radius: 50%; background: {ACCENT}; color: #fff; font-family: "Space Grotesk"; font-weight: 700; font-size: 38px; align-items: center; justify-content: center; margin-right: 18px; flex: none; }}
h2 {{ margin: 0; font-size: 56px; line-height: 1.08; display: flex; align-items: center; letter-spacing: -0.01em; }}
.lead {{ margin: 0; color: {INK2}; }}
.card {{ background: {PANEL}; border-radius: 18px; padding: 26px 30px; }}
.card h3 {{ margin: 0 0 14px; font-size: 30px; font-family: "Space Grotesk"; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 26px; font-size: 23px; color: {INK2}; margin-bottom: 14px; }}
.lg {{ display: inline-flex; align-items: center; gap: 10px; }}
.lg i {{ width: 26px; height: 16px; border-radius: 4px; display: inline-block; }}
ul.find {{ margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 16px; }}
ul.find li {{ padding-left: 34px; position: relative; }}
ul.find li::before {{ content: ""; position: absolute; left: 0; top: 14px; width: 14px; height: 14px; border-radius: 50%; background: {ACCENT}; }}
ul.find b {{ font-weight: 700; }}
.note {{ font-size: 21px; color: {MUTED}; margin: 6px 0 0; }}
.panels {{ display: flex; gap: 10px; justify-content: space-between; }}
.ptitle {{ font-size: 27px; font-weight: 600; margin: 0 0 0 60px; white-space: nowrap; }}
.pcount {{ font-size: 21px; color: {MUTED}; margin: 0 0 0 60px; }}
table.dis {{ width: 100%; border-collapse: collapse; font-size: 25px; }}
table.dis th {{ text-align: left; font-size: 21px; color: {MUTED}; font-weight: 600; padding: 0 12px 8px 0; border-bottom: 2px solid {RULE}; }}
table.dis td {{ padding: 12px 12px 12px 0; border-bottom: 1.5px solid {RULE}; vertical-align: top; }}
table.dis td.auc {{ white-space: nowrap; font-weight: 700; font-family: "Space Grotesk"; font-size: 30px; }}
.callouts {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.callout {{ display: grid; grid-template-columns: 330px 1fr; gap: 26px; align-items: center; }}
.big {{ font-size: 62px; font-weight: 700; color: {ACCENT}; line-height: 1; letter-spacing: -0.02em; }}
.shot {{ width: 100%; height: 640px; object-fit: cover; object-position: top; border-radius: 14px; border: 2px solid {RULE}; display: block; }}
.flow {{ display: flex; align-items: center; gap: 10px; }}
.step {{ flex: 1 1 0; background: #fff; border: 2px solid {RULE}; border-radius: 14px; padding: 18px 14px; font-size: 23px; line-height: 1.25; text-align: center; min-height: 132px; display: flex; flex-direction: column; justify-content: center; }}
.step b {{ display: block; font-size: 24px; margin-bottom: 4px; }}
.step.hl {{ background: {ACCENT}; color: #fff; border-color: {ACCENT}; }}
.arrow {{ flex: none; width: 28px; height: 28px; }}
footer {{ background: {ACCENT}; color: #fff; display: flex; align-items: center; justify-content: space-between; padding: 0 96px; gap: 48px; }}
footer .msg {{ font-family: "Space Grotesk"; font-size: 46px; font-weight: 600; line-height: 1.15; max-width: 2700px; }}
footer .src {{ font-size: 22px; color: #cfe7e3; text-align: right; }}
</style></head>
<body><div class="poster">

<header>
  <div>
    <h1>Explaining Anonymity:<span class="t2">Adversarial Attacks and Science Communication</span></h1>
    <div class="who">Ananya Jogalekar · Explainable AI, Duke University</div>
  </div>
  <div class="qr"><div class="box" id="qr" aria-label="QR code linking to the website"></div>
    <div class="txt"><b>Try the site</b>Pick any of the {F['n_models']} models and see its explanations, attacks and checklist.</div></div>
</header>

<main>
<section>
  <h2><span class="num">1</span>What the explanations told us</h2>
  <p class="lead">We generated 5,000 accounts where we <b>know</b> what makes an account fake: a formula over 12 account stats, bursty posting, and giveaway words like “giveaway” and “sheep”. So every explanation can be graded right or wrong, not just plausible. Each model is explained with the two methods most used for its type (e.g. SHAP and LIME).</p>
  <div class="card">
    <h3>Where SHAP puts the credit (account-stat models)</h3>
    {credit_chart()}
    <p class="note">Share of SHAP credit over 150 test accounts. Good models put most of it on the 6 features that really drive the label; anomaly detectors waste half on noise and decoys.</p>
  </div>
  <ul class="find">
    <li><b>Right features, missed subtlety.</b> LightGBM gives {F['shap_share']} of its credit to the real signal, but only {F['inter_shap']} to the burstiness × account-age interaction, which carries {F['inter_true']} of it.</li>
    <li><b>Activity models ignore timing.</b> Shuffling all 168 hours of a week changes the LSTM's AUC by {F['timing']}: it counts posts, not rhythm.</li>
    <li><b>A shortcut fooled the text models.</b> #trending marked fakes only in training: the transformer fell from {F['tr_val']} to {F['tr_test']} AUC, and its explanations showed why.</li>
    <li><b>Methods often disagree.</b> LIME gives different answers on re-runs; one explanation alone can mislead. {F['good']} of {F['n_models']} models pass all our checks.</li>
  </ul>
</section>

<section>
  <h2><span class="num">2</span>When the fakes fight back</h2>
  <p class="lead">A fresh population of accounts whose bots and trolls disguise themselves, at mild and strong strength. Real users never change, #trending is gone, and every disguise is something a real operator can do cheaply.</p>
  <div class="card">
    <h3>Detection (AUC) as the disguise gets stronger</h3>
    {attack_chart()}
    <p class="note">Combined disguise per input type; grey lines are the other models of that type. AUC 1.0 = perfect; the darker line at 0.5 = a coin flip.</p>
  </div>
  <table class="dis">
    <tr><th>Input</th><th>One cheap disguise</th><th>AUC</th></tr>
    <tr><td>Account stats</td><td>post as often as a real user</td><td class="auc">{F['lgb_none']} → {F['lgb_post']}</td></tr>
    <tr><td>Hourly activity</td><td>post only as much as a real user</td><td class="auc">{F['lstm_none']} → {F['lstm_vol']}</td></tr>
    <tr><td>Post text</td><td>misspell the giveaway words (g1veaway)</td><td class="auc">{F['tr_none']} → {F['tr_disg']}</td></tr>
  </table>
  <div class="callouts">
    <div class="callout"><div class="big">{F['lgb_caught0']}→{F['lgb_caught1']}</div><div>Fakes LightGBM catches once they copy real users' habits.</div></div>
    <div class="callout"><div class="big">{F['shift_lo']}–{F['shift_hi']}%</div><div>of SHAP's credit moves while the account-stat models fail: <b>the explanations don't warn you</b>.</div></div>
    <div class="callout"><div class="big">{F['iso_mild']}</div><div>Isolation forest under a <i>mild</i> disguise ({F['iso_strong']} under strong): half-disguised fakes look more typical than real users.</div></div>
  </div>
</section>

<section>
  <h2><span class="num">3</span>From notebook to pipeline</h2>
  <p class="lead">Explainability results usually live in notebooks, one model at a time: hard to compare, and never turned into a decision. A developer about to deploy a detector needs a <b>verdict</b> and a <b>to-do list</b>, not 40 SHAP plots.</p>
  <img class="shot" src="{screenshot}" alt="A model page on the Explaining Anonymity website">
  <ul class="find">
    <li><b>A verdict from published rules.</b> “Good” only if accurate, stable on new data <i>and</i> its explanations find the real signal. Same rules for every model; nothing hand-written.</li>
    <li><b>Strengths, weaknesses and a checklist</b>, each item saying why it is there.</li>
    <li><b>Under attack:</b> a strength slider shows detection and explanations as fakes hide better.</li>
  </ul>
  <div class="card">
    <h3>A step in the deployment pipeline</h3>
    <div class="flow">
      <div class="step"><b>Train</b>new model version</div>
      <svg class="arrow" viewBox="0 0 28 28" aria-hidden="true"><path d="M4 14h18M15 7l7 7-7 7" fill="none" stroke="{MUTED}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="step hl"><b>Answer-key test</b>are explanations right?</div>
      <svg class="arrow" viewBox="0 0 28 28" aria-hidden="true"><path d="M4 14h18M15 7l7 7-7 7" fill="none" stroke="{MUTED}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="step hl"><b>Attack test</b>does it survive disguises?</div>
      <svg class="arrow" viewBox="0 0 28 28" aria-hidden="true"><path d="M4 14h18M15 7l7 7-7 7" fill="none" stroke="{MUTED}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="step"><b>Verdict + checklist</b>deploy or fix</div>
    </div>
    <p class="note">Already code: data generators, evaluation scripts, rules with 66 tests, and a static site rebuilt in seconds. Prototype today covers {F['n_models']} models on synthetic data; next, letting teams plug in their own model and data.</p>
  </div>
</section>
</main>

<footer>
  <div class="msg">Accurate and well-explained is not the same as safe to deploy: every model fell to cheap disguises, and explanations rarely warned us.</div>
  <div class="src">{REPO_URL}<br>All data synthetic · results reproducible from seeds</div>
</footer>
</div>
<script>
  (function () {{
    var qr = qrcode(0, "M"); qr.addData({json.dumps(SITE_URL)}); qr.make();
    document.getElementById("qr").innerHTML = qr.createSvgTag({{ cellSize: 4, margin: 0, scalable: true }});
  }})();
</script>
</body></html>
"""


def main():
    out_html = HERE / "poster.html"
    out_pdf = HERE / "poster.pdf"
    out_html.write_text(build_html())
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    "--allow-file-access-from-files", f"--print-to-pdf={out_pdf}", out_html.as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {out_html.relative_to(ROOT)} and {out_pdf.relative_to(ROOT)} ({W_IN}x{H_IN} in)")


if __name__ == "__main__":
    main()
