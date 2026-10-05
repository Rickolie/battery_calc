// Runs the Python analysis (battery_calc package) in Pyodide, off the UI thread.
const PYODIDE_VERSION = "0.27.2";
const LOCAL = "pyodide/";
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
let py = null;
let manifest = null;

function post(type, data) { self.postMessage({ type, ...data }); }

async function boot() {
  let indexURL = LOCAL;
  try {
    importScripts(LOCAL + "pyodide.js");          // vendored by the deploy workflow
  } catch (e) {
    indexURL = CDN;                               // local development fallback
    importScripts(CDN + "pyodide.js");
  }
  post("status", { text: "Starting Python…" });
  py = await loadPyodide({ indexURL });
  post("status", { text: "Loading numpy, pandas, matplotlib, scipy…" });
  await py.loadPackage(["numpy", "pandas", "matplotlib", "pyyaml", "scipy"]);
  manifest = await (await fetch("manifest.json", { cache: "no-cache" })).json();
  post("status", { text: "Loading site data…" });
  const files = [...manifest.package, manifest.config, ...manifest.data];
  await Promise.all(files.map(async (f) => {
    const r = await fetch(f, { cache: "no-cache" });
    if (!r.ok) throw new Error(`could not load ${f}`);
    writeFile("/site/" + f, new Uint8Array(await r.arrayBuffer()));
  }));
  py.runPython(`
import os, sys
os.chdir("/site")
sys.path.insert(0, "/site")
os.environ["MPLBACKEND"] = "Agg"
from battery_calc import web_api
`);
  post("ready", { manifest });
}

function writeFile(path, data) {
  const parts = path.split("/").filter(Boolean);
  let dir = "";
  for (const p of parts.slice(0, -1)) {
    dir += "/" + p;
    try { py.FS.mkdir(dir); } catch (e) { /* exists */ }
  }
  py.FS.writeFile(path, data);
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.cmd === "validate") {
      writeFile("/site/uploads/" + m.name, m.text);
      const api = py.pyimport("battery_calc.web_api");
      post("validated", { result: JSON.parse(api.validate_p1("uploads/" + m.name)) });
    } else if (m.cmd === "upload") {
      writeFile("/site/uploads/" + m.name, m.text);
    } else if (m.cmd === "run") {
      const api = py.pyimport("battery_calc.web_api");
      const cb = (json) => post("section", { section: JSON.parse(json) });
      const out = api.run(JSON.stringify(m.settings), cb);
      post("done", { result: JSON.parse(out) });
    }
  } catch (e) {
    post("error", { text: String(e && e.message ? e.message : e) });
  }
};

boot().catch((e) => post("error", { text: "Could not start: " + e }));
