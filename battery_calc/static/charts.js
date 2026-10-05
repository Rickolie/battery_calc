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
  window.renderInteractiveCharts = function (root) {
    if (typeof uPlot === "undefined") return;
    (root || document).querySelectorAll("div.ichart").forEach(build);
  };
  if (document.readyState !== "loading") window.renderInteractiveCharts();
  else document.addEventListener("DOMContentLoaded", () => window.renderInteractiveCharts());
})();
