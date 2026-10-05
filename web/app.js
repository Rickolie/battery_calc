// UI for the static site. The analysis itself runs in worker.js (Pyodide).
const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js");
const state = { useRicks: true, p1Ok: false, csv: {}, sections: [] };

function status(text, cls = "muted") { const s = $("status"); s.textContent = text; s.className = cls; }
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "status") status(m.text);
  else if (m.type === "ready") onReady(m.manifest);
  else if (m.type === "validated") onValidated(m.result);
  else if (m.type === "section") onSection(m.section);
  else if (m.type === "done") onDone(m.result);
  else if (m.type === "error") { status("Error: " + m.text, "err"); $("run").disabled = false; }
};

function onReady(manifest) {
  status("Ready.", "ok");
  if (manifest.online_updated_at) $("dataDate").textContent = `Online data last updated ${manifest.online_updated_at}.`;
  const sel = $("connection");
  for (const c of manifest.connections) {
    const o = document.createElement("option");
    o.value = c; o.textContent = c.replace("x", " × ") + " A";
    if (c === manifest.default_connection) o.selected = true;
    sel.appendChild(o);
  }
  for (const b of manifest.batteries || []) {
    if (!b.price) continue;
    const o = document.createElement("option");
    o.value = b.id; o.textContent = `${b.name} (€ ${Math.round(Number(b.price))})`;
    $("chosen").appendChild(o);
  }
  if (manifest.blackfriday) {
    if (manifest.blackfriday.discount_nl != null) $("bfnl").value = Math.round(manifest.blackfriday.discount_nl * 100);
    if (manifest.blackfriday.discount_de != null) $("bfde").value = Math.round(manifest.blackfriday.discount_de * 100);
  }
  if (manifest.kiln) {
    if (manifest.kiln.firing_hours != null) $("kh").value = manifest.kiln.firing_hours;
    if (manifest.kiln.avg_duty != null) $("kd").value = Math.round(manifest.kiln.avg_duty * 100);
  }
  $("useRick").disabled = !manifest.ricks_data;
  if (!manifest.ricks_data) $("useRick").title = "Rick's data is not published with this site";
  $("useMine").disabled = false;
}

$("useRick").onclick = () => { state.useRicks = true; hide("step2"); show("step3"); };
$("useMine").onclick = () => { state.useRicks = false; show("step2"); hide("step3"); };

async function readText(input) { return input.files[0] ? await input.files[0].text() : null; }

$("p1").onchange = async () => {
  const text = await readText($("p1"));
  if (!text) return;
  $("p1check").textContent = "Checking…";
  $("toSettings").disabled = true;
  worker.postMessage({ cmd: "validate", name: "p1.csv", text });
};

$("pv").onchange = async () => {
  const text = await readText($("pv"));
  if (text) worker.postMessage({ cmd: "upload", name: "pv.csv", text });
};

function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]); }

function onValidated(r) {
  const el = $("p1check");
  if (!r.ok) { el.innerHTML = `<span class="err">Cannot use this file: ${esc(r.error)}</span>`; return; }
  const miss = r.missing_months.length ? `<span class="warn">Missing months (will be filled synthetically and flagged): ${esc(r.missing_months.join(", "))}</span>` : `<span class="ok">No missing months.</span>`;
  el.innerHTML = `<ul>
    <li>Data ${esc(r.first)} → ${esc(r.last)} (${r.rows} rows); profile year ${esc(r.window)}</li>
    <li>${miss}${r.low_confidence ? ' <span class="warn">More than 3 months missing: low confidence.</span>' : ""}</li>
    <li>Duplicates ${r.duplicates}, counter resets ${r.resets}, short gaps interpolated ${r.interpolated_gaps}, long gaps ${r.long_gaps.length}${r.long_gaps.length ? ": " + esc(r.long_gaps.join("; ")) : ""}</li>
    <li>Yearly import ${r.import_kwh} kWh, export ${r.export_kwh} kWh (real data only: ${r.import_real_kwh} / ${r.export_real_kwh})</li></ul>`;
  state.p1Ok = true;
  $("toSettings").disabled = false;
}

$("toSettings").onclick = () => show("step3");

function num(id) { const v = parseFloat($(id).value.replace(",", ".")); return isNaN(v) ? null : v; }
function pct(id) { const v = num(id); return v == null ? null : v / 100; }

function contractForm() {
  const v = (id) => $(id).value.trim().replace(",", ".");
  const f = { supplier: $("c_supplier").value.trim(), price_t1: v("c_t1"), price_t2: v("c_t2"), price_single: v("c_single"),
    fixed_eur_year: v("c_fixed"), feed_in: v("c_feed"), feed_in_cost_kwh: v("c_feedcost"),
    feed_in_cost_tiers: $("c_tiers").value.trim(), duration_months: v("c_months"), type: "fixed" };
  const any = ["price_t1", "price_t2", "price_single"].some((k) => f[k]);
  return any ? f : null;
}

// Every section the analysis produces, in order; shown as a live contents list.
const SECTIONS = [
  ["advice", "Advice – which battery, which contract, and why"], ["data", "1. Your data"],
  ["power", "2. Power profile"], ["current", "3. Current contract"],
  ["contracts", "4. Cheapest contract without a battery"], ["batteries", "5. Battery options"],
  ["breakeven", "6. Break-even price per battery"],
  ["payback", "7. Payback per battery size and strategy (slowest step)"],
  ["breakeven_recomputed", "8. Break-even with simulated cycles"],
  ["sanity", "9. Battery over the year – check per size"],
  ["kiln", "10. Pottery kiln on free power"],
];

function renderToc() {
  const ul = document.createElement("ul");
  ul.id = "toc";
  for (const [id, title] of SECTIONS) {
    const li = document.createElement("li");
    li.id = "toc-" + id;
    li.innerHTML = `<span class="muted">${esc(title)} – computing…</span>`;
    ul.appendChild(li);
  }
  $("toc").replaceWith(ul);
}

function tocDone(sec) {
  const li = document.getElementById("toc-" + sec.id);
  if (li) li.innerHTML = `<a href="#${sec.id}">${esc(sec.title)}</a> ✓`;
}

$("run").onclick = () => {
  $("run").disabled = true;
  state.csv = {}; state.sections = [];
  $("results").innerHTML = ""; $("downloads").innerHTML = "";
  show("step4");
  renderToc();
  status("Running… sections appear as they are computed.");
  const settings = {
    use_ricks: state.useRicks, label: state.useRicks ? "Rick's data" : "Uploaded data",
    p1_path: "uploads/p1.csv", pv_path: $("pv").files[0] ? "uploads/pv.csv" : null,
    contract: state.useRicks ? null : contractForm(),
    connection: $("connection").value, margin: parseFloat($("margin").value), price_variant: $("variant").value,
    feed_in_2030: parseFloat($("fi2030").value), quick: $("quick").value === "1",
    battery_set: $("bset").value, chosen_battery: $("chosen").value,
    blackfriday: { discount_nl: pct("bfnl"), discount_de: pct("bfde"), estimates: $("bfest").checked },
    kiln: { firing_hours: num("kh"), avg_duty: pct("kd") },
  };
  worker.postMessage({ cmd: "run", settings });
};

function onSection(sec) {
  const div = document.createElement("div");
  div.innerHTML = sec.html;
  const links = document.createElement("p");
  links.className = "downloads";
  for (const [name, csv] of Object.entries(sec.csv)) {
    state.csv[name] = csv;
    links.appendChild(downloadLink(name, csv, "text/csv"));
  }
  (div.querySelector("section .secbody") || div.querySelector("section")).appendChild(links);
  // The recommendation goes to the top; everything else in order of arrival.
  if (sec.id === "advice") $("results").prepend(div); else $("results").appendChild(div);
  if (window.renderInteractiveCharts) window.renderInteractiveCharts(div);
  tocDone(sec);
  const next = SECTIONS.find(([id]) => !document.getElementById(id) && id !== "advice");
  status(`Computed: ${sec.title}` + (next ? ` – now working on: ${next[1]}` : ""));
}

function downloadLink(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.textContent = name;
  return a;
}

function onDone(r) {
  status("Done.", "ok");
  $("run").disabled = false;
  const d = $("downloads");
  d.appendChild(downloadLink("report.html", r.html, "text/html"));
  d.appendChild(downloadLink("report.md", r.markdown, "text/markdown"));
}
