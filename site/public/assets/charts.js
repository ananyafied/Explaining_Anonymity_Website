// Small, dependency-free chart components. All text goes in with textContent (data is never parsed as HTML).

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("--")) e.style.setProperty(k, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

export const pct = (x, d = 0) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
export const num = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

// ------------------------------------------------------------------ tooltip (one per page)
let tipEl;
function tip() {
  if (!tipEl) { tipEl = el("div", { class: "tip", role: "tooltip", hidden: true }); document.body.append(tipEl); }
  return tipEl;
}
export function showTip(x, y, title, rows) {
  const t = tip();
  t.replaceChildren(el("div", { class: "t-title", text: title }));
  for (const r of rows) {
    const k = el("span", { class: "k" }, r.color ? el("i", { "--c": r.color }) : null, r.name);
    t.append(el("div", { class: "t-row" }, k, el("b", { text: r.value })));
  }
  t.hidden = false;
  const w = t.offsetWidth, h = t.offsetHeight;
  const left = Math.min(Math.max(8, x + 14), window.innerWidth - w - 8);
  const top = y + h + 20 > window.innerHeight ? y - h - 12 : y + 16;
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}
export function hideTip() { if (tipEl) tipEl.hidden = true; }
function tipAt(node, title, rows) {
  const r = node.getBoundingClientRect();
  showTip(r.left + Math.min(r.width / 2, 160), r.top, title, rows);
}
/** Hover + keyboard focus show the same tooltip. */
function bindTip(node, title, rowsFn) {
  node.tabIndex = 0;
  node.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, title, rowsFn()));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => tipAt(node, title, rowsFn()));
  node.addEventListener("blur", hideTip);
}

// ------------------------------------------------------------------ legend + table view
export function legend(items) {
  return el("div", { class: "legend" }, items.map((i) =>
    el("span", {}, el("i", { class: `key ${i.kind || ""}`, "--c": i.color }), i.label)));
}

export function tableView(headers, rows, summary = "Show as a table") {
  const table = el("table", {},
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", { text: h })))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c) => el("td", { text: c }))))));
  return el("details", { class: "table-view" }, el("summary", { text: summary }), el("div", { class: "table-scroll" }, table));
}

// ------------------------------------------------------------------ horizontal bars (+ optional truth tick)
/**
 * rows: [{label, tags: [str], values: {seriesKey: number}, truth: number|null}]
 * series: [{key, name, color}]
 */
export function hbars({ rows, series, truthName = "True share", fmt = pct, max }) {
  const top = max ?? Math.max(...rows.flatMap((r) => [...series.map((s) => r.values[s.key] ?? 0), r.truth ?? 0]));
  const scale = (v) => `${Math.max(0, (100 * (v ?? 0)) / (top || 1))}%`;
  const grid = el("div", { class: "hbars", role: "list" });
  for (const r of rows) {
    const lab = el("div", { class: "lab" }, r.label, ...(r.tags || []).map((t) => el("span", { class: "tag", text: t })));
    const track = el("div", { class: "track" },
      series.map((s) => el("div", { class: "bar", "--c": s.color, style: { width: scale(r.values[s.key]) } })),
      r.truth != null ? el("div", { class: "tick", style: { left: scale(r.truth) } }) : null);
    const val = el("div", { class: "val", text: fmt(r.values[series[0].key]) });
    const row = el("div", { class: "row", role: "listitem" }, lab, track, val);
    const rowsFn = () => [
      ...series.map((s) => ({ name: s.name, color: s.color, value: fmt(r.values[s.key]) })),
      ...(r.truth != null ? [{ name: truthName, color: "var(--truth)", value: fmt(r.truth) }] : []),
    ];
    bindTip(track, r.label, rowsFn);
    track.setAttribute("aria-label", `${r.label}: ${rowsFn().map((x) => `${x.name} ${x.value}`).join(", ")}`);
    grid.append(row);
  }
  const ticks = [0, top / 2, top].map((v) => el("span", { text: fmt(v) }));
  const axis = el("div", { class: "axis-row" }, el("span"), el("div", { class: "ax" }, ticks), el("span"));
  return el("div", {}, grid, axis);
}

// ------------------------------------------------------------------ signed bars: left = toward real, right = toward fake
/**
 * rows: [{label, tags, cells: {colKey: number}, note: str}]
 * cols: [{key, name, color}]   each column is scaled to its own largest |value|.
 */
export function signedTable({ rows, cols, firstHeader = "Feature", noteHeader = null, fmt = (v) => num(v, 3) }) {
  const maxAbs = Object.fromEntries(cols.map((c) => [c.key, Math.max(1e-12, ...rows.map((r) => Math.abs(r.cells[c.key] ?? 0)))]));
  const head = el("tr", {}, el("th", { text: firstHeader }), noteHeader ? el("th", { text: noteHeader }) : null,
    cols.map((c) => el("th", {}, el("span", { class: "legend", style: { margin: 0 } }, el("span", {}, el("i", { class: "key", "--c": c.color }), c.name)))));
  const body = rows.map((r) => {
    const tr = el("tr", {}, el("td", {}, r.label, ...(r.tags || []).map((t) => el("span", { class: "tag", style: { marginLeft: "4px" }, text: t }))),
      noteHeader ? el("td", { class: "num", text: r.note ?? "" }) : null);
    for (const c of cols) {
      const v = r.cells[c.key];
      const w = `${(50 * Math.abs(v ?? 0)) / maxAbs[c.key]}%`;
      const bar = el("div", { class: "sbar" }, v == null ? null : el("i", { class: v >= 0 ? "pos" : "neg", "--c": c.color, style: { width: w } }));
      bindTip(bar, r.label, () => [{ name: c.name, color: c.color, value: v == null ? "–" : `${fmt(v)} (${v >= 0 ? "toward fake" : "toward real"})` }]);
      bar.setAttribute("aria-label", `${c.name}: ${v == null ? "no value" : fmt(v)}`);
      tr.append(el("td", {}, bar));
    }
    return tr;
  });
  const dirRow = el("tr", {}, el("td"), noteHeader ? el("td") : null,
    cols.map(() => el("td", {}, el("div", { class: "dir" }, el("span", { text: "← real" }), el("span", { text: "fake →" })))));
  const colgroup = el("colgroup", {}, el("col", { class: "c-label" }), noteHeader ? el("col", { class: "c-note" }) : null, cols.map(() => el("col")));
  return el("div", { class: "signed-scroll" }, el("table", { class: "signed" }, colgroup, el("thead", {}, head), el("tbody", {}, body, dirRow)));
}

// ------------------------------------------------------------------ hourly activity: counts + planted hours, then one panel per method
export function hourChart({ counts, planted, series }) {
  const N = counts.length;
  const wrap = el("div", { class: "hours-wrap", tabindex: 0, "aria-label": "Hourly chart. Use the left and right arrow keys to step through the hours." });
  const cross = el("div", { class: "crosshair", hidden: true });

  // panel 1: posts per hour, planted hours shaded
  const H1 = 60, maxC = Math.max(1, ...counts);
  const s1 = svg("svg", { class: "hours", viewBox: `0 0 ${N} ${H1}`, preserveAspectRatio: "none", height: 90, "aria-hidden": "true" });
  planted.forEach((p, i) => { if (p) s1.append(svg("rect", { x: i, y: 0, width: 1, height: H1, style: "fill: var(--s3-wash)" })); });
  counts.forEach((c, i) => {
    if (c > 0) { const h = (H1 - 2) * (c / maxC); s1.append(svg("rect", { x: i + 0.15, y: H1 - h, width: 0.7, height: h, style: "fill: var(--muted); opacity: .75" })); }
  });
  s1.append(svg("line", { x1: 0, x2: N, y1: H1 - 0.5, y2: H1 - 0.5, style: "stroke: var(--axis)", "vector-effect": "non-scaling-stroke", "stroke-width": 1 }));

  const panels = [el("div", { class: "panel-label" }, "Posts per hour", el("span", { class: "tag", text: `max ${maxC}` })), s1];

  // one panel per method: |attribution| scaled to the method's own peak
  const H2 = 40;
  for (const s of series) {
    const vals = s.values.map((v) => Math.abs(v ?? 0));
    const peak = Math.max(1e-12, ...vals);
    const pts = vals.map((v, i) => [i + 0.5, H2 - 1 - (H2 - 3) * (v / peak)]);
    const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("");
    const g = svg("svg", { class: "hours", viewBox: `0 0 ${N} ${H2}`, preserveAspectRatio: "none", height: 56, "aria-hidden": "true" });
    planted.forEach((p, i) => { if (p) g.append(svg("rect", { x: i, y: 0, width: 1, height: H2, style: "fill: var(--s3-wash)" })); });
    g.append(svg("path", { d: `${d}L${N - 0.5},${H2 - 1}L0.5,${H2 - 1}Z`, style: `fill: ${s.color}; opacity: .12` }));
    g.append(svg("path", { d, style: `fill: none; stroke: ${s.color}`, "stroke-width": 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-linecap": "round" }));
    g.append(svg("line", { x1: 0, x2: N, y1: H2 - 0.5, y2: H2 - 0.5, style: "stroke: var(--axis)", "vector-effect": "non-scaling-stroke", "stroke-width": 1 }));
    panels.push(el("div", { class: "panel-label" }, el("i", { class: "key line", "--c": s.color }), `${s.name}: credit per hour`), g);
  }
  const axis = el("div", { class: "hour-axis" }, Array.from({ length: 8 }, (_, d) => el("span", { text: d < 7 ? `Day ${d + 1}` : "" })));
  const inner = el("div", { style: { position: "relative" } }, ...panels, cross);
  wrap.append(inner, axis);

  let hour = null;
  const show = (h, x, y) => {
    hour = Math.max(0, Math.min(N - 1, h));
    const r = inner.getBoundingClientRect();
    cross.hidden = false;
    cross.style.left = `${((hour + 0.5) / N) * r.width}px`;
    const rows = [
      { name: "Posts", value: String(counts[hour]) },
      { name: "Planted pattern", value: planted[hour] ? "yes" : "no" },
      ...series.map((s) => ({ name: s.name, color: s.color, value: num(Math.abs(s.values[hour] ?? 0), 2) })),
    ];
    showTip(x ?? r.left + ((hour + 0.5) / N) * r.width, y ?? r.top, `Day ${Math.floor(hour / 24) + 1}, ${String(hour % 24).padStart(2, "0")}:00`, rows);
  };
  inner.addEventListener("pointermove", (e) => {
    const r = inner.getBoundingClientRect();
    show(Math.floor(((e.clientX - r.left) / r.width) * N), e.clientX, e.clientY);
  });
  inner.addEventListener("pointerleave", () => { cross.hidden = true; hideTip(); });
  wrap.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    show((hour ?? -1) + (e.key === "ArrowRight" ? 1 : -1));
  });
  wrap.addEventListener("blur", () => { cross.hidden = true; hideTip(); });
  return wrap;
}

// ------------------------------------------------------------------ tall segmented bars (part-to-whole)
/** rows: [{who, parts: [{name, value, color, darkInk}]}] — values are shares summing to ~1. */
export function segBars(rows) {
  const box = el("div");
  for (const r of rows) {
    const bar = el("div", { class: "segbar", role: "img", "aria-label": `${r.who}: ${r.parts.map((p) => `${p.name} ${pct(p.value)}`).join(", ")}` });
    for (const p of r.parts) {
      if (!p.value) continue;
      const seg = el("div", { class: p.darkInk ? "dark-ink" : "", "--c": p.color, style: { flex: `${p.value} 1 0` } });
      if (p.value >= 0.12) seg.textContent = pct(p.value);
      bindTip(seg, `${r.who} · ${p.name}`, () => [{ name: "Share of credit", color: p.color, value: pct(p.value) }]);
      bar.append(seg);
    }
    box.append(el("div", { class: "segrow" }, el("div", { class: "who", text: r.who }), bar));
  }
  return box;
}

// ------------------------------------------------------------------ segmented control
export function segTabs(options, current, onChange) {
  const box = el("div", { class: "seg-tabs", role: "group" });
  const btns = options.map((o) => el("button", { type: "button", "aria-pressed": String(o.key === current), text: o.label, onclick: () => {
    btns.forEach((b) => b.setAttribute("aria-pressed", "false"));
    btns[options.indexOf(o)].setAttribute("aria-pressed", "true");
    onChange(o.key);
  } }));
  box.append(...btns);
  return box;
}
