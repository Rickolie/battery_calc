// Interactive charts for the report and the website (uPlot, MIT licence).
// Each chart is a <div class="ichart"> followed by <script type="application/json">.
// Drag horizontally to zoom, double-click to reset, click a legend item to hide a series.
(function () {
  function barsPath() {
    return uPlot.paths && uPlot.paths.bars ? uPlot.paths.bars({ size: [0.8, 40] }) : null;
  }
  function fmt(v, unit) {
    if (v == null) return "–";
    const a = Math.abs(v);
    return (a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2)) + (unit ? " " + unit : "");
  }
  function build(div) {
    if (div.dataset.done) return;
    const json = div.nextElementSibling;
    if (!json || json.type !== "application/json") return;
    const spec = JSON.parse(json.textContent);
    div.dataset.done = "1";
    if (spec.kind) { svgChart(div, spec); return; }
    const data = [spec.x];
    const series = [{ value: (u, ts) => ts == null ? "–" : new Date(ts * 1000).toLocaleString("nl-NL",
      spec.daily ? { dateStyle: "medium" } : { dateStyle: "short", timeStyle: "short" }) }];
    const scales = { x: { time: true } };
    let hasY2 = false;
    for (const s of spec.series) {
      data.push(s.values);
      const scale = s.scale || "y";
      if (scale === "y2") hasY2 = true;
      const unit = scale === "y2" ? spec.y2 : spec.y;
      const conf = { label: s.label, stroke: s.color, width: s.type === "bar" ? 0 : (s.width || 1.5), scale,
                     value: (u, v) => fmt(v, unit) };
      if (s.type === "bar") { conf.fill = s.color + "cc"; conf.paths = barsPath(); conf.points = { show: false }; }
      if (s.type === "area") { conf.fill = s.color + "33"; }
      if (s.dash) conf.dash = s.dash;
      series.push(conf);
    }
    const axes = [{ stroke: "#888", grid: { stroke: "#8883" } },
                  { label: spec.y, stroke: "#888", grid: { stroke: "#8883" }, size: 60 }];
    if (hasY2) { scales.y2 = {}; axes.push({ scale: "y2", side: 1, label: spec.y2, stroke: "#888", grid: { show: false }, size: 60 }); }
    const opts = {
      title: spec.title, width: Math.max(320, div.clientWidth || 800), height: spec.height || 320,
      series, scales, axes, cursor: { drag: { x: true, y: false }, focus: { prox: 30 } }, legend: { live: true },
    };
    const u = new uPlot(opts, data, div);
    if (window.ResizeObserver) {
      new ResizeObserver(() => u.setSize({ width: Math.max(320, div.clientWidth), height: opts.height })).observe(div);
    }
  }

  // ---------------------------------------------------------------- SVG charts
  // Small categorical / numeric charts (bars, scatter, a few lines) with a hover
  // tooltip, a legend and selective direct labels. Labels are set with textContent.
  const NS = "http://www.w3.org/2000/svg";
  const INK = "#0b0b0b", INK2 = "#52514e", GRID = "#e4e3df", MUTED = "#c9c8c3";
  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, opts) {
    const t = el("text", Object.assign({ x, y, fill: INK2, "font-size": 11 }, opts || {}), parent);
    t.textContent = s;
    return t;
  }
  function niceTicks(lo, hi, n) {
    if (hi === lo) { hi = lo + 1; }
    const raw = (hi - lo) / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [], end = Math.ceil(hi / step - 1e-9) * step;
    for (let v = Math.floor(lo / step + 1e-9) * step; v <= end + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function money(v, unit) {
    if (v == null || isNaN(v)) return "–";
    const a = Math.abs(v);
    const s = a >= 1000 ? Math.round(v).toLocaleString("nl-NL") : a >= 10 ? v.toFixed(0) : v.toFixed(1);
    return unit === "€" ? "€" + s : s + (unit ? " " + unit : "");
  }
  function tooltip(div) {
    let tip = div.querySelector(".svgtip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "svgtip";
      tip.style.cssText = "position:absolute;pointer-events:none;background:#fff;border:1px solid #ddd;" +
        "border-radius:6px;padding:6px 8px;font-size:12px;color:#0b0b0b;box-shadow:0 2px 8px #0002;display:none;" +
        "white-space:nowrap;z-index:5";
      div.appendChild(tip);
    }
    return {
      show(x, y, title, rows) {
        tip.textContent = "";
        const h = document.createElement("div"); h.style.cssText = "font-weight:600;margin-bottom:3px";
        h.textContent = title; tip.appendChild(h);
        for (const [color, value, label] of rows) {
          const r = document.createElement("div");
          const k = document.createElement("span");
          k.style.cssText = `display:inline-block;width:12px;height:2px;background:${color};vertical-align:middle;margin-right:6px`;
          const v = document.createElement("b"); v.textContent = value;
          const l = document.createElement("span"); l.style.color = INK2; l.textContent = " " + label;
          r.append(k, v, l); tip.appendChild(r);
        }
        tip.style.display = "block";
        const w = div.clientWidth, tw = tip.offsetWidth;
        tip.style.left = Math.min(Math.max(0, x + 12), w - tw - 4) + "px";
        tip.style.top = Math.max(0, y - 10) + "px";
      },
      hide() { tip.style.display = "none"; },
    };
  }
  function legend(div, items, shape) {
    const lg = document.createElement("div");
    lg.style.cssText = "display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:" + INK2 + ";margin:2px 0 6px";
    for (const it of items) {
      const s = document.createElement("span");
      const k = document.createElement("span");
      k.style.cssText = shape === "line"
        ? `display:inline-block;width:14px;height:2px;background:${it.color};vertical-align:middle;margin-right:5px`
        : `display:inline-block;width:10px;height:10px;border-radius:2px;background:${it.color};vertical-align:middle;margin-right:5px`;
      s.append(k, document.createTextNode(it.label));
      lg.appendChild(s);
    }
    div.appendChild(lg);
  }
  function frame(div, spec) {
    div.style.position = "relative";
    div.style.padding = "8px 10px";
    if (spec.title) {
      const h = document.createElement("div");
      h.style.cssText = "font-weight:600;font-size:14px;color:" + INK + ";margin-bottom:2px";
      h.textContent = spec.title; div.appendChild(h);
    }
    if (spec.subtitle) {
      const h = document.createElement("div");
      h.style.cssText = "font-size:12px;color:" + INK2 + ";margin-bottom:4px";
      h.textContent = spec.subtitle; div.appendChild(h);
    }
  }
  function axes(svg, W, H, m, ys, yScale, yLabel, unit) {
    const step = ys.length > 1 ? Math.abs(ys[1] - ys[0]) : 1;
    const dec = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
    for (const v of ys) {
      const y = yScale(v);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: v === 0 ? MUTED : GRID, "stroke-width": 1 }, svg);
      const n = v.toLocaleString("nl-NL", { minimumFractionDigits: dec, maximumFractionDigits: dec });
      txt(svg, m.l - 6, y + 4, unit === "€" ? "€" + n : unit === "yr" ? n + " yr" : n, { "text-anchor": "end" });
    }
    if (yLabel) txt(svg, 12, m.t + (H - m.t - m.b) / 2, yLabel,
      { transform: `rotate(-90 12 ${m.t + (H - m.t - m.b) / 2})`, "text-anchor": "middle" });
  }

  function svgChart(div, spec) {
    frame(div, spec);
    const W = Math.max(320, div.clientWidth - 20 || 800), H = spec.height || 320;
    if (spec.kind === "bars") return barChart(div, spec, W, H);
    if (spec.kind === "scatter") return scatterChart(div, spec, W, H);
    if (spec.kind === "lines") return lineChart(div, spec, W, H);
  }

  // Stacked (positive up, negative down) or grouped columns per category, with an
  // optional total marker + value label per category.
  function barChart(div, spec, W, H) {
    if (spec.series.length > 1) legend(div, spec.series.concat(spec.total ? [{ label: spec.total.label, color: INK }] : []), "box");
    const svg = el("svg", { width: W, height: H, role: "img" }, div);
    const m = { l: 64, r: 12, t: 16, b: spec.rotate ? 70 : 34 };
    const cats = spec.categories, n = cats.length, k = spec.series.length;
    let lo = 0, hi = 0;
    cats.forEach((_, i) => {
      if (spec.stacked) {
        let p = 0, q = 0;
        for (const s of spec.series) { const v = s.values[i] || 0; if (v > 0) p += v; else q += v; }
        hi = Math.max(hi, p); lo = Math.min(lo, q);
      } else for (const s of spec.series) { const v = s.values[i] || 0; hi = Math.max(hi, v); lo = Math.min(lo, v); }
      if (spec.total) { hi = Math.max(hi, spec.total.values[i]); lo = Math.min(lo, spec.total.values[i]); }
    });
    if (spec.ydomain) { lo = Math.min(lo, spec.ydomain[0]); hi = Math.max(hi, spec.ydomain[1]); }
    const ys = niceTicks(lo, hi, 5);
    const y0 = ys[0], y1 = ys[ys.length - 1];
    const yScale = (v) => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
    axes(svg, W, H, m, ys, yScale, spec.yLabel, spec.unit);
    const band = (W - m.l - m.r) / n;
    const tip = tooltip(div);
    cats.forEach((cat, i) => {
      const cx = m.l + band * (i + 0.5);
      const bw = spec.stacked ? Math.min(24 * 1.6, band * 0.6) : Math.min(24, (band * 0.8) / k);
      let pos = 0, neg = 0;
      spec.series.forEach((s, j) => {
        const v = s.values[i] || 0;
        if (!v) return;
        let x, yTop, yBot;
        if (spec.stacked) {
          x = cx - bw / 2;
          if (v > 0) { yBot = yScale(pos); pos += v; yTop = yScale(pos); }
          else { yTop = yScale(neg); neg += v; yBot = yScale(neg); }
        } else {
          x = cx - (k * bw) / 2 + j * bw;
          yTop = yScale(Math.max(v, 0)); yBot = yScale(Math.min(v, 0));
        }
        const h = Math.max(0, yBot - yTop - (spec.stacked ? 2 : 0));
        el("rect", { x: x + (spec.stacked ? 0 : 1), y: yTop + (spec.stacked && v > 0 ? 2 : 0), width: Math.max(1, bw - (spec.stacked ? 0 : 2)),
                     height: h, rx: 2, fill: s.color }, svg);
      });
      if (spec.total) {
        const v = spec.total.values[i], y = yScale(v);
        el("circle", { cx, cy: y, r: 4.5, fill: INK, stroke: "#fff", "stroke-width": 2 }, svg);
        txt(svg, cx + 9, y + 4, money(v, spec.unit), { fill: INK, "font-weight": 600 });
      }
      const lab = txt(svg, cx, H - m.b + 16, cat, { "text-anchor": spec.rotate ? "end" : "middle", fill: INK2 });
      if (spec.rotate) lab.setAttribute("transform", `rotate(-35 ${cx} ${H - m.b + 16})`);
      const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent" }, svg);
      const rows = spec.series.map((s) => [s.color, money(s.values[i], spec.unit), s.label]);
      if (spec.total) rows.unshift([INK, money(spec.total.values[i], spec.unit), spec.total.label]);
      hit.addEventListener("pointermove", (e) => {
        const r = div.getBoundingClientRect();
        tip.show(e.clientX - r.left, e.clientY - r.top, cat, rows);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
    });
  }

  function scatterChart(div, spec, W, H) {
    if (spec.legend) legend(div, spec.legend, "box");
    const svg = el("svg", { width: W, height: H, role: "img" }, div);
    const m = { l: 56, r: 16, t: 14, b: 40 };
    const P = spec.points;
    const xs = niceTicks(0, Math.max(...P.map((p) => p.x)), 6);
    const yv = P.map((p) => p.y).filter((v) => isFinite(v));
    const ys = niceTicks(0, Math.max(...yv), 5);
    const xScale = (v) => m.l + (W - m.l - m.r) * (v - xs[0]) / (xs[xs.length - 1] - xs[0]);
    const yScale = (v) => m.t + (H - m.t - m.b) * (1 - (v - ys[0]) / (ys[ys.length - 1] - ys[0]));
    axes(svg, W, H, m, ys, yScale, spec.yLabel, spec.yUnit);
    for (const v of xs) txt(svg, xScale(v), H - m.b + 16, String(v), { "text-anchor": "middle" });
    if (spec.xLabel) txt(svg, m.l + (W - m.l - m.r) / 2, H - 6, spec.xLabel, { "text-anchor": "middle" });
    const tip = tooltip(div);
    const order = P.slice().sort((a, b) => (a.strong ? 1 : 0) - (b.strong ? 1 : 0));
    for (const p of order) {
      if (!isFinite(p.y)) continue;
      const x = xScale(p.x), y = yScale(p.y);
      el("circle", { cx: x, cy: y, r: p.strong ? 6 : 4.5, fill: p.color, stroke: "#fff", "stroke-width": 2 }, svg);
      if (p.strong && p.short) {
        const left = x > W * 0.65;     // keep labels inside the plot near the right edge
        txt(svg, left ? x - 9 : x + 9, y - 8, p.short, { fill: INK, "font-weight": 600, "text-anchor": left ? "end" : "start" });
      }
      const hit = el("circle", { cx: x, cy: y, r: 12, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => {
        const r = div.getBoundingClientRect();
        tip.show(e.clientX - r.left, e.clientY - r.top, p.label, p.rows || []);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
    }
  }

  // A few numeric-x lines; emphasised series are coloured and end-labelled,
  // the rest are thin grey context. Crosshair snaps to the nearest x.
  function lineChart(div, spec, W, H) {
    const strong = spec.series.filter((s) => !s.context);
    const leg = strong.map((s) => ({ label: s.label, color: s.color }));
    if (spec.series.some((s) => s.context)) leg.push({ label: spec.contextLabel || "other options", color: MUTED });
    if (leg.length > 1) legend(div, leg, "line");
    const svg = el("svg", { width: W, height: H, role: "img" }, div);
    const m = { l: 64, r: spec.endLabels ? 120 : 16, t: 14, b: 36 };
    const X = spec.x;
    let lo = Infinity, hi = -Infinity;
    for (const s of spec.series) for (const v of s.values) if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (spec.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const ys = niceTicks(lo, hi, 5);
    const x0 = Math.min(...X), x1 = Math.max(...X);
    const xScale = (v) => m.l + (W - m.l - m.r) * (v - x0) / ((x1 - x0) || 1);
    const yScale = (v) => m.t + (H - m.t - m.b) * (1 - (v - ys[0]) / (ys[ys.length - 1] - ys[0]));
    axes(svg, W, H, m, ys, yScale, spec.yLabel, spec.unit);
    const xt = spec.xTicks || X;
    const every = Math.max(1, Math.ceil(xt.length / 12));
    xt.forEach((v, i) => { if (i % every === 0) txt(svg, xScale(v), H - m.b + 16, spec.xFormat === "year" ? String(Math.round(v)) : String(v), { "text-anchor": "middle" }); });
    if (spec.xLabel) txt(svg, m.l + (W - m.l - m.r) / 2, H - 4, spec.xLabel, { "text-anchor": "middle" });
    const draw = (s, color, width) => {
      const pts = X.map((x, i) => [x, s.values[i]]).filter(([, v]) => v != null && isFinite(v));
      if (!pts.length) return;
      el("path", { d: pts.map(([x, v], i) => (i ? "L" : "M") + xScale(x).toFixed(1) + " " + yScale(v).toFixed(1)).join(""),
                   fill: "none", stroke: color, "stroke-width": width, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      return pts;
    };
    for (const s of spec.series) if (s.context) draw(s, MUTED, 1);
    for (const s of strong) {
      const pts = draw(s, s.color, 2);
      if (!pts) continue;
      if (spec.markers !== false) for (const [x, v] of pts) el("circle", { cx: xScale(x), cy: yScale(v), r: 3, fill: s.color }, svg);
      if (spec.endLabels) {
        const [x, v] = pts[pts.length - 1];
        txt(svg, xScale(x) + 6, yScale(v) + 4, s.short || s.label, { fill: INK });
      }
    }
    (spec.points || []).forEach((mk, i) => {
      el("circle", { cx: xScale(mk.x), cy: yScale(mk.y), r: 5, fill: mk.color || INK, stroke: "#fff", "stroke-width": 2 }, svg);
      // alternate labels above / below so nearby points stay readable
      if (mk.label) txt(svg, xScale(mk.x), yScale(mk.y) + (i % 2 ? 20 : -9), mk.label,
                        { fill: INK, "text-anchor": "middle", "font-weight": 600 });
    });
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: MUTED, "stroke-width": 1, visibility: "hidden" }, svg);
    const tip = tooltip(div);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let i = 0, best = Infinity;
      X.forEach((x, j) => { const d = Math.abs(xScale(x) - px); if (d < best) { best = d; i = j; } });
      cross.setAttribute("x1", xScale(X[i])); cross.setAttribute("x2", xScale(X[i])); cross.setAttribute("visibility", "visible");
      let rows = strong.map((s) => [s.color, money(s.values[i], spec.unit), s.label]);
      const ctx = spec.series.filter((s) => s.context && s.values[i] != null);
      if (ctx.length) {   // the context line nearest the pointer
        let near = null, nd = Infinity;
        for (const s of ctx) { const d = Math.abs(yScale(s.values[i]) - py); if (d < nd) { nd = d; near = s; } }
        if (near && nd < 30) rows.push([MUTED, money(near.values[i], spec.unit), near.label]);
      }
      const t = spec.xNames ? spec.xNames[i] : (spec.xFormat === "year" ? String(Math.round(X[i])) : String(X[i]));
      const d = div.getBoundingClientRect();
      tip.show(e.clientX - d.left, e.clientY - d.top, t, rows);
    });
    hit.addEventListener("pointerleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); });
  }

  // Charts inside a closed <details> have no width yet: build them when opened.
  window.renderInteractiveCharts = function (root) {
    if (typeof uPlot === "undefined") return;
    (root || document).querySelectorAll("div.ichart").forEach((div) => {
      if (div.offsetParent !== null) build(div);
    });
  };
  document.addEventListener("toggle", (e) => {
    if (e.target.open) window.renderInteractiveCharts(e.target);
  }, true);
  // Expand / collapse every section and table.
  window.setAllDetails = function (open) {
    document.querySelectorAll("details.sec, details.tblw, details.more").forEach((d) => { d.open = open; });
    if (open) window.renderInteractiveCharts();
  };
  // A link to a section (contents list) opens it.
  document.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("a[href^='#']");
    if (!a) return;
    const target = document.getElementById(a.getAttribute("href").slice(1));
    const d = target && target.querySelector("details.sec");
    if (d) d.open = true;
  });
  if (document.readyState !== "loading") window.renderInteractiveCharts();
  else document.addEventListener("DOMContentLoaded", () => window.renderInteractiveCharts());
})();
