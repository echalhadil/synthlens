const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const status = document.getElementById("status");
const result = document.getElementById("result");
const verdictLabel = document.getElementById("verdict-label");
const meterFill = document.getElementById("meter-fill");
const prob = document.getElementById("prob");
const thumbs = document.getElementById("thumbs");
const tbody = document.querySelector("#heuristics tbody");
const viz = document.getElementById("viz");
const headerInfo = document.getElementById("header-info");

function colorForVerdict(v) {
  if (v.startsWith("Likely Real")) return "var(--good)";
  if (v.startsWith("Likely AI")) return "var(--bad)";
  return "var(--warn)";
}

function fmtDetails(d) {
  return Object.entries(d)
    .filter(([k]) => k !== "exif" && k !== "chunks")
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

function humanSize(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(n) / Math.log(1024));
  return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${u[i]}`;
}

function kvTable(obj) {
  const rows = Object.entries(obj).map(([k, v]) =>
    `<tr><td class="k">${k}</td><td>${v}</td></tr>`).join("");
  return `<table class="kvtable"><tbody>${rows}</tbody></table>`;
}

function renderHeader(meta) {
  const d = meta.details;
  let html = "";
  html += `<div class="hrow"><span class="k">format</span><span class="v">${d.format}</span></div>`;
  html += `<div class="hrow"><span class="k">dimensions</span><span class="v">${d.dimensions}</span></div>`;
  html += `<div class="hrow"><span class="k">file size</span><span class="v">${humanSize(d.file_size)}</span></div>`;
  const exif = d.exif || {};
  const chunks = d.chunks || {};
  html += `<div class="group">EXIF (${Object.keys(exif).length} tags)</div>`;
  html += Object.keys(exif).length ? kvTable(exif) : `<div class="empty">no EXIF metadata</div>`;
  html += `<div class="group">embedded chunks</div>`;
  html += Object.keys(chunks).length ? kvTable(chunks) : `<div class="empty">no text chunks</div>`;
  headerInfo.innerHTML = html;
}

async function upload(file) {
  result.classList.add("hidden");
  status.classList.remove("hidden");
  status.textContent = `analyzing ${file.name}…`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/detect", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    render(data, file);
  } catch (e) {
    status.textContent = `error: ${e.message}`;
  }
}

function render(data, file) {
  status.classList.add("hidden");
  result.classList.remove("hidden");

  verdictLabel.textContent = data.verdict;
  verdictLabel.style.color = colorForVerdict(data.verdict);
  meterFill.style.width = `${Math.round(data.probability * 100)}%`;
  meterFill.style.background = colorForVerdict(data.verdict);
  prob.textContent = `probability AI: ${(data.probability * 100).toFixed(0)}%`;

  thumbs.innerHTML = "";
  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  img.alt = "uploaded";
  thumbs.appendChild(img);

  if (data.heuristics.metadata) renderHeader(data.heuristics.metadata);

  tbody.innerHTML = "";
  const entries = Object.entries(data.heuristics).sort(
    (a, b) => b[1].score * b[1].weight - a[1].score * a[1].weight
  );
  for (const [name, h] of entries) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${h.score.toFixed(2)}</td><td>${h.weight.toFixed(2)}</td><td>${(h.score * h.weight).toFixed(3)}</td><td class="details">${fmtDetails(h.details)}</td>`;
    tbody.appendChild(tr);
  }

  viz.innerHTML = "";
  for (const [name, h] of Object.entries(data.heuristics)) {
    if (!h.viz) continue;
    for (const [vname, b64] of Object.entries(h.viz)) {
      if (!b64) continue;
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `<h4>${name} · ${vname}</h4><img src="data:image/png;base64,${b64}" alt="${name} ${vname}" />`;
      viz.appendChild(card);
    }
  }
}

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && upload(fileInput.files[0]));

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});