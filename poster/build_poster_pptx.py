"""Build the Explaining Anonymity poster as an editable PowerPoint file: poster/poster.pptx.

    <python with python-pptx> poster/build_poster_pptx.py

One 36 x 24 in slide mirroring poster/poster.pdf: editable text boxes, native PowerPoint charts and
table, and two pictures (the QR code and a website screenshot). Numbers come from the same saved
results as the PDF (via build_poster.facts and friends). Fonts are Arial so the file looks the same
on any computer that opens or prints it.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build_poster as BP  # noqa: E402  (data and copy shared with the PDF)

FONT = "Arial"
rgb = lambda h: RGBColor.from_string(h.lstrip("#"))  # noqa: E731
INK, INK2, MUTED, RULE, PAPER, PANEL = (rgb(c) for c in (BP.INK, BP.INK2, BP.MUTED, BP.RULE, BP.PAPER, BP.PANEL))
ACCENT, BLUE, ORANGE, AQUA, NOISE = (rgb(c) for c in (BP.ACCENT, BP.BLUE, BP.ORANGE, BP.AQUA, BP.NOISE))
WHITE, GREY_LINE, HEAD_SUB, HEAD_WHO = rgb("#ffffff"), rgb("#c4c1b8"), rgb("#e9e6de"), rgb("#b9b6ad")

BODY = Pt(22)
F = BP.facts()


# ------------------------------------------------------------------ helpers
def rect(slide, x, y, w, h, fill, name, shape=MSO_SHAPE.RECTANGLE, radius=None, line=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb, s.line.width = line
    if radius is not None:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, name, size=BODY, color=INK, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, spacing=1.15, after=0):
    """paras: list of paragraphs; each a str or a list of runs (text, {bold, italic, color, size})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(after)
        bullet = False
        if isinstance(para, dict):  # {"bullet": True, "runs": [...]}
            bullet, para = para.get("bullet", False), para["runs"]
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = o.get("size", size)
            f.bold = o.get("bold", False)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
        if bullet:
            _bullet(p)
    return tb


def _bullet(p):
    """A real PowerPoint bullet (teal dot, hanging indent), not a typed character."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.36)))
    pPr.set("indent", str(-Inches(0.36)))
    for tag in ("a:buClr", "a:buSzPct", "a:buFont", "a:buChar"):
        el = pPr.find(qn(tag))
        if el is not None:
            pPr.remove(el)
    clr = pPr.makeelement(qn("a:buClr"), {})
    srgb = clr.makeelement(qn("a:srgbClr"), {"val": BP.ACCENT.lstrip("#")})
    clr.append(srgb)
    sz = pPr.makeelement(qn("a:buSzPct"), {"val": "110000"})
    font = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    char = pPr.makeelement(qn("a:buChar"), {"char": "•"})
    for el in (clr, sz, font, char):
        pPr.append(el)


def picture(slide, path, x, y, w, h, name, alt, border=None):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    if border:
        pic.line.color.rgb, pic.line.width = border
    return pic


def heading(slide, x, y, num, title):
    c = rect(slide, x, y + 0.04, 0.66, 0.66, ACCENT, f"Section {num} number", shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(num)
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(28), True, WHITE
    text(slide, x + 0.88, y, 9.9, 0.75, [title], f"Section {num} heading", size=Pt(40), color=INK, anchor=MSO_ANCHOR.MIDDLE)


def style_axis_text(axis, size=Pt(17), color=INK2):
    tl = axis.tick_labels
    tl.font.size, tl.font.name, tl.font.color.rgb = size, FONT, color


# ------------------------------------------------------------------ charts
def credit_chart(slide, x, y, w, h):
    import pandas as pd

    df = pd.read_csv(ROOT / "outputs" / "step2_explain" / "tabular" / "credit_by_feature_group.csv")
    df = df[(df["method"] == "SHAP") & df["model"].isin(BP.NAMES)].rename(
        columns={"informative (6)": "inf", "proxy posts_per_week": "proxy", "decoys (2)": "dec", "noise (3)": "noise"})
    df = df.sort_values("inf", ascending=True)  # bar charts draw the first category at the bottom
    cd = CategoryChartData()
    cd.categories = [BP.NAMES[k] for k in df["model"]]
    groups = [("inf", "Real signal (6 features)", BLUE), ("proxy", "Near-copy", ORANGE), ("dec", "Decoys", AQUA), ("noise", "Noise", NOISE)]
    for col, name, _ in groups:
        cd.add_series(name, [int(round(100 * float(v))) for v in df[col]])  # whole percentages
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = "Chart: SHAP credit by feature group"
    ch = gf.chart
    ch.font.name, ch.font.size = FONT, Pt(17)
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 38, 100
    for s, (_, _, color) in zip(plot.series, groups):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = color
        s.format.line.color.rgb = PANEL  # thin panel-coloured gap between segments
        s.format.line.width = Pt(2)
    dl = plot.series[0].data_labels
    dl.show_value, dl.number_format, dl.number_format_is_linked = True, '0"%"', False
    dl.position = XL_LABEL_POSITION.INSIDE_BASE
    dl.font.size, dl.font.bold, dl.font.color.rgb, dl.font.name = Pt(17), True, WHITE, FONT
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    ca = ch.category_axis
    ca.format.line.fill.background()
    style_axis_text(ca, Pt(18), INK)
    ch.has_legend = False  # legend drawn as shapes above the chart, so it looks the same in every app
    return gf


def legend_row(slide, x, y, items, name):
    """Swatch + label pairs as ordinary shapes and text."""
    cx = x
    for i, (label, color) in enumerate(items):
        rect(slide, cx, y + 0.08, 0.32, 0.2, color, f"{name} swatch {i + 1}", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
        width = 0.15 + 0.105 * len(label)
        text(slide, cx + 0.42, y, width, 0.36, [label], f"{name} label {i + 1}", size=Pt(17), color=INK2, anchor=MSO_ANCHOR.MIDDLE)
        cx += 0.42 + width + 0.3


def attack_chart(slide, x, y, w, h, fam, title, hero):
    keys = [k for k in BP.NAMES if BP.FAMILY_OF[k] == fam]
    keys = sorted(keys, key=lambda k: k == hero)  # hero drawn last (on top)
    cd = CategoryChartData()
    cd.categories = ["None", "Mild", "Strong"]
    for k in keys:
        a = BP.model_json(k)["attack"]
        c = a["results"][a["combined"]]
        cd.add_series(BP.NAMES[k], [round(a["results"]["none"]["auc"], 3), round(c["mild"]["auc"], 3), round(c["strong"]["auc"], 3)])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = f"Chart: AUC under disguise, {title}"
    ch = gf.chart
    ch.font.name, ch.font.size = FONT, Pt(16)
    ch.has_legend = False
    ch.has_title = False  # title and highlighted model are text boxes above the chart
    for s, k in zip(ch.plots[0].series, keys):
        s.smooth = False
        ln = s.format.line
        if k == hero:
            ln.color.rgb, ln.width = BLUE, Pt(4.5)
            s.marker.style, s.marker.size = XL_MARKER_STYLE.CIRCLE, 11
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = BLUE
            s.marker.format.line.color.rgb = PANEL
        else:
            ln.color.rgb, ln.width = GREY_LINE, Pt(2.25)
            s.marker.style = XL_MARKER_STYLE.NONE
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1, 0.5
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RULE
    va.format.line.fill.background()
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.0", False
    style_axis_text(va, Pt(16), MUTED)
    ca = ch.category_axis
    ca.format.line.color.rgb = RULE
    style_axis_text(ca, Pt(16), INK2)
    text(slide, x + 0.1, y - 0.95, w - 0.1, 0.9, [
        [(title, {"size": Pt(19), "bold": True}), (f" ({len(keys)})", {"size": Pt(17), "color": MUTED})],
        [(BP.NAMES[hero], {"size": Pt(16), "bold": True, "color": BLUE}), (" in blue", {"size": Pt(16), "color": MUTED})],
    ], f"Chart title: {title}", spacing=1.05)
    return gf


def table(slide, x, y, w, rows):
    shape = slide.shapes.add_table(len(rows), 3, Inches(x), Inches(y), Inches(w), Inches(0.58 * len(rows)))
    shape.name = "Table: one cheap disguise per input"
    tbl = shape.table
    tbl.columns[0].width = Inches(2.3)
    tbl.columns[1].width = Inches(w - 2.3 - 2.05)
    tbl.columns[2].width = Inches(2.05)
    tbl.first_row = True
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(0.58)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if i == 0 else (WHITE if i % 2 else PAPER)
            cell.margin_left, cell.margin_right = Inches(0.12), Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = val
            r = tf.paragraphs[0].runs[0]
            r.font.name = FONT
            r.font.color.rgb = MUTED if i == 0 else INK
            r.font.size = Pt(16) if i == 0 else (Pt(22) if j == 2 else Pt(19))
            r.font.bold = (i == 0) or (j == 2)
    return shape


# ------------------------------------------------------------------ the slide
def build(out: Path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(BP.W_IN), Inches(BP.H_IN)
    prs.core_properties.title = "Explaining Anonymity: Adversarial Attacks and Science Communication"
    prs.core_properties.author = "Ananya Jogalekar"
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER

    # header
    rect(slide, 0, 0, 36, 3.85, INK, "Header band")
    text(slide, 1.0, 0.62, 25.5, 2.4, [
        [("Explaining Anonymity:", {"size": Pt(78), "bold": True, "color": WHITE})],
        [("Adversarial Attacks and Science Communication", {"size": Pt(50), "bold": True, "color": HEAD_SUB})],
    ], "Title", spacing=1.0, after=4)
    text(slide, 1.0, 2.83, 20, 0.5, ["Ananya Jogalekar · Explainable AI, Duke University"], "Author", size=Pt(24), color=HEAD_WHO)
    rect(slide, 27.55, 0.5, 2.85, 2.85, WHITE, "QR code backing", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    picture(slide, HERE / "qr.png", 27.67, 0.62, 2.61, 2.61, "QR code", f"QR code linking to {BP.SITE_URL}")
    text(slide, 30.75, 1.0, 4.25, 2.0, [
        [("Try the site", {"size": Pt(26), "bold": True, "color": WHITE})],
        [(f"Pick any of the {F['n_models']} models and see its explanations, attacks and checklist.", {"size": Pt(19), "color": HEAD_SUB})],
    ], "Try the site", after=6)

    # columns
    cw, gap, top = 10.83, 0.75, 4.42
    xs = [1.0, 1.0 + cw + gap, 1.0 + 2 * (cw + gap)]
    B = lambda t: (t, {"bold": True})  # noqa: E731
    N = lambda t: (t, {})  # noqa: E731
    I = lambda t: (t, {"italic": True})  # noqa: E731

    # ---- column 1
    x = xs[0]
    heading(slide, x, top, 1, "What the explanations told us")
    text(slide, x, top + 1.0, cw, 2.15, [[
        N("We generated 5,000 accounts where we "), B("know"),
        N(" what makes an account fake: a formula over 12 account stats, bursty posting, and giveaway words like “giveaway” and “sheep”. "
          "So every explanation can be graded right or wrong, not just plausible. Each model is explained with the two methods most used for its type (e.g. SHAP and LIME)."),
    ]], "Section 1 lead", color=INK2)
    rect(slide, x, top + 3.3, cw, 7.55, PANEL, "Card: SHAP credit", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.025)
    text(slide, x + 0.3, top + 3.5, cw - 0.6, 0.5, [[B("Where SHAP puts the credit (account-stat models)")]], "Card title: SHAP credit", size=Pt(23))
    legend_row(slide, x + 0.3, top + 4.1, [("Real signal (6 features)", BLUE), ("Near-copy", ORANGE), ("Decoys", AQUA), ("Noise", NOISE)], "Credit legend")
    credit_chart(slide, x + 0.15, top + 4.55, cw - 0.4, 5.35)
    text(slide, x + 0.3, top + 9.95, cw - 0.6, 0.75, [
        "Share of SHAP credit over 150 test accounts. Good models put most of it on the 6 features that really drive the label; anomaly detectors waste half on noise and decoys."],
        "Card note: SHAP credit", size=Pt(16), color=MUTED)
    text(slide, x, top + 11.15, cw, 6.2, [
        {"bullet": True, "runs": [B("Right features, missed subtlety. "), N(f"LightGBM gives {F['shap_share']} of its credit to the real signal, but only {F['inter_shap']} to the burstiness × account-age interaction, which carries {F['inter_true']} of it.")]},
        {"bullet": True, "runs": [B("Activity models ignore timing. "), N(f"Shuffling all 168 hours of a week changes the LSTM's AUC by {F['timing']}: it counts posts, not rhythm.")]},
        {"bullet": True, "runs": [B("A shortcut fooled the text models. "), N(f"#trending marked fakes only in training: the transformer fell from {F['tr_val']} to {F['tr_test']} AUC, and its explanations showed why.")]},
        {"bullet": True, "runs": [B("Methods often disagree. "), N(f"LIME gives different answers on re-runs; one explanation alone can mislead. {F['good']} of {F['n_models']} models pass all our checks.")]},
    ], "Section 1 findings", after=12)

    # ---- column 2
    x = xs[1]
    heading(slide, x, top, 2, "When the fakes fight back")
    text(slide, x, top + 1.0, cw, 1.8, [
        "A fresh population of accounts whose bots and trolls disguise themselves, at mild and strong strength. Real users never change, #trending is gone, and every disguise is something a real operator can do cheaply."],
        "Section 2 lead", color=INK2)
    rect(slide, x, top + 2.95, cw, 6.55, PANEL, "Card: AUC under disguise", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.025)
    text(slide, x + 0.3, top + 3.15, cw - 0.6, 0.5, [[B("Detection (AUC) as the disguise gets stronger")]], "Card title: AUC under disguise", size=Pt(23))
    pw = (cw - 0.5) / 3
    for i, (fam, title, hero) in enumerate([("tabular", "Account stats", "lightgbm"), ("sequences", "Hourly activity", "lstm"), ("text", "Post text", "transformer")]):
        attack_chart(slide, x + 0.2 + i * pw, top + 4.7, pw - 0.05, 3.95, fam, title, hero)
    text(slide, x + 0.3, top + 8.7, cw - 0.6, 0.7, [
        "Combined disguise per input type; grey lines are the other models of that type. AUC 1.0 = perfect; 0.5 = a coin flip."],
        "Card note: AUC under disguise", size=Pt(16), color=MUTED)
    table(slide, x, top + 9.8, cw, [
        ["Input", "One cheap disguise", "AUC"],
        ["Account stats", "post as often as a real user", f"{F['lgb_none']} → {F['lgb_post']}"],
        ["Hourly activity", "post only as much as a real user", f"{F['lstm_none']} → {F['lstm_vol']}"],
        ["Post text", "misspell the giveaway words (g1veaway)", f"{F['tr_none']} → {F['tr_disg']}"],
    ])
    callouts = [
        (f"{F['lgb_caught0']}→{F['lgb_caught1']}", [N("Fakes LightGBM catches once they copy real users' habits.")]),
        (f"{F['shift_lo']}–{F['shift_hi']}%", [N("of SHAP's credit moves while the account-stat models fail: "), B("the explanations don't warn you.")]),
        (F["iso_mild"], [N("Isolation forest under a "), I("mild"), N(f" disguise ({F['iso_strong']} under strong): half-disguised fakes look more typical than real users.")]),
    ]
    cy = top + 12.45
    for i, (big, runs) in enumerate(callouts):
        text(slide, x, cy, 3.6, 1.2, [[(big, {"size": Pt(46), "bold": True, "color": ACCENT})]], f"Callout {i + 1} number", anchor=MSO_ANCHOR.MIDDLE)
        text(slide, x + 3.8, cy, cw - 3.8, 1.2, [runs], f"Callout {i + 1} text", anchor=MSO_ANCHOR.MIDDLE)
        cy += 1.55

    # ---- column 3
    x = xs[2]
    heading(slide, x, top, 3, "From notebook to pipeline")
    text(slide, x, top + 1.0, cw, 1.8, [[
        N("Explainability results usually live in notebooks, one model at a time: hard to compare, and never turned into a decision. A developer about to deploy a detector needs a "),
        B("verdict"), N(" and a "), B("to-do list"), N(", not 40 SHAP plots."),
    ]], "Section 3 lead", color=INK2)
    picture(slide, HERE / "site_crop.png", x, top + 2.95, cw, cw * 670 / 1100, "Website screenshot",
            "A model page on the Explaining Anonymity website: LightGBM's verdict, scores and checks", border=(RULE, Pt(2)))
    text(slide, x, top + 9.75, cw, 3.4, [
        {"bullet": True, "runs": [B("A verdict from published rules. "), N("“Good” only if accurate, stable on new data "), I("and"), N(" its explanations find the real signal. Same rules for every model; nothing hand-written.")]},
        {"bullet": True, "runs": [B("Strengths, weaknesses and a checklist"), N(", each item saying why it is there.")]},
        {"bullet": True, "runs": [B("Under attack: "), N("a strength slider shows detection and explanations as fakes hide better.")]},
    ], "Section 3 points", after=12)
    cy = top + 13.35
    rect(slide, x, cy, cw, 4.1, PANEL, "Card: pipeline", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03)
    text(slide, x + 0.3, cy + 0.2, cw - 0.6, 0.5, [[B("A step in the deployment pipeline")]], "Card title: pipeline", size=Pt(23))
    steps = [("Train", "new model version", False), ("Answer-key test", "are explanations right?", True),
             ("Attack test", "does it survive disguises?", True), ("Verdict + checklist", "deploy or fix", False)]
    sw, arrow = 2.18, 0.42
    sx = x + 0.3
    for i, (head, sub, hl) in enumerate(steps):
        box = rect(slide, sx, cy + 0.9, sw, 1.55, ACCENT if hl else WHITE, f"Pipeline step {i + 1}", shape=MSO_SHAPE.ROUNDED_RECTANGLE,
                   radius=0.1, line=None if hl else (RULE, Pt(2)))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for j, (t, sz, b) in enumerate(((head, Pt(18), True), (sub, Pt(16), False))):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = t
            r.font.name, r.font.size, r.font.bold = FONT, sz, b
            r.font.color.rgb = WHITE if hl else INK
        sx += sw
        if i < len(steps) - 1:
            a = rect(slide, sx + 0.07, cy + 1.5, arrow - 0.14, 0.34, MUTED, f"Pipeline arrow {i + 1}", shape=MSO_SHAPE.RIGHT_ARROW)
            sx += arrow
    text(slide, x + 0.3, cy + 2.65, cw - 0.6, 1.3, [
        f"Already code: data generators, evaluation scripts, rules with 66 tests, and a static site rebuilt in seconds. "
        f"Prototype today covers {F['n_models']} models on synthetic data; next, letting teams plug in their own model and data."],
        "Card note: pipeline", size=Pt(16), color=MUTED)

    # footer
    rect(slide, 0, 22.33, 36, 1.67, ACCENT, "Footer band")
    text(slide, 1.0, 22.33, 27.3, 1.67, [
        "Accurate and well-explained is not the same as safe to deploy: every model fell to cheap disguises, and explanations rarely warned us."],
        "Takeaway", size=Pt(32), color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 28.6, 22.33, 6.4, 1.67, [BP.REPO_URL, "All data synthetic · results reproducible from seeds"],
         "Source", size=Pt(16), color=rgb("#cfe7e3"), anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)

    prs.save(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    build(HERE / "poster.pptx")
