/* EngageWatch dashboard: talks to the FastAPI service on the same origin. */
"use strict";

const CLASS_COLORS = {
  engaged: "#1b9e77", confused: "#3b82c4", frustrated: "#7b61c9",
  bored: "#d98b1f", drowsy: "#c2553d", looking_away: "#b0447a",
};
const GROUP = { bored: "Not engaged", confused: "Engaged", drowsy: "Not engaged",
                engaged: "Engaged", frustrated: "Engaged", looking_away: "Not engaged" };
let CLASSES = ["bored", "confused", "drowsy", "engaged", "frustrated", "looking_away"];

const $ = (id) => document.getElementById(id);
const nice = (s) => String(s).replace(/_/g, " ");
const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const num = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const charts = {};

if (window.Chart) {
  Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
  Chart.defaults.font.size = 12.5;
  Chart.defaults.color = "#4b5263";
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { /* empty body */ }
  if (!response.ok) {
    const detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : response.statusText;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

function drawChart(id, config) {
  if (!window.Chart) return null;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), config);
  return charts[id];
}

function metricTiles(container, items) {
  container.innerHTML = items.map(([label, value]) =>
    `<div class="metric"><div class="m-label">${esc(label)}</div><div class="m-value">${esc(value)}</div></div>`).join("");
}

function fillSelect(select) {
  select.innerHTML = CLASSES.map((c) => `<option value="${c}">${nice(c)} (${GROUP[c]})</option>`).join("");
}

/* ------------------------------------------------------------------ nav */
const loaders = {};
document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + button.dataset.view));
    if (loaders[button.dataset.view]) loaders[button.dataset.view]();
  });
});

/* ------------------------------------------------------------- overview */
const health = { ok: 0, total: 0, series: [], labels: [] };
let uptimeBase = null, uptimeAt = null;

function formatUptime(seconds) {
  const s = Math.floor(seconds);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  const pad = (n) => String(n).padStart(2, "0");
  return (d ? d + "d " : "") + `${pad(h)}:${pad(m)}:${pad(s % 60)}`;
}

function setStatus(state, text) {
  for (const el of [$("side-dot"), $("ov-status").querySelector(".dot")]) el.className = "dot " + state;
  $("side-status-text").textContent = text;
  $("ov-status").lastChild.textContent = " " + text;
}

async function pingHealth() {
  const started = performance.now();
  health.total += 1;
  let ms = null;
  try {
    const body = await api("/health", { cache: "no-store" });
    ms = performance.now() - started;
    health.ok += 1;
    uptimeBase = body.uptime_seconds; uptimeAt = Date.now();
    setStatus(body.model_loaded ? "ok" : "warn", body.model_loaded ? "Online" : "Online, no model");
  } catch (_) {
    setStatus("down", "Unreachable");
    uptimeBase = null;
  }
  health.series.push(ms == null ? null : Math.round(ms * 10) / 10);
  health.labels.push(new Date().toLocaleTimeString());
  if (health.series.length > 60) { health.series.shift(); health.labels.shift(); }
  $("ov-availability").textContent = pct(health.ok / health.total, 1);
  $("ov-pings").textContent = `${health.ok} of ${health.total} health checks passed`;
  if (charts["chart-health"]) {
    charts["chart-health"].data.labels = health.labels;
    charts["chart-health"].data.datasets[0].data = health.series;
    charts["chart-health"].update("none");
  }
}

async function loadStatus() {
  let s;
  try { s = await api("/status", { cache: "no-store" }); } catch (_) { return; }
  CLASSES = s.classes || CLASSES;
  $("ov-instance").textContent = "instance " + s.instance;
  $("ov-started").textContent = "started " + s.started_at.replace("T", " ").replace("+00:00", " UTC");
  const m = s.model;
  $("ov-version").textContent = m.loaded ? "v" + m.version : "not loaded";
  $("ov-trained").textContent = m.trained_at
    ? `${m.kind === "retrain" ? "retrained" : "trained"} ${m.trained_at.replace("T", " ").replace("+00:00", " UTC")}` : "–";
  $("ov-arch").textContent = m.architecture ? `${m.architecture}. Trained on ${m.train_images} images.` : "No model file found.";
  const t = m.test_metrics || {};
  metricTiles($("ov-metrics"), [
    ["Accuracy", pct(t.accuracy)], ["Loss", num(t.loss)], ["Precision", pct(t.precision_macro)],
    ["Recall", pct(t.recall_macro)], ["F1 score", pct(t.f1_macro)], ["ROC-AUC", num(t.roc_auc_ovr)],
    ["Engaged vs not", pct(t.engaged_vs_not_accuracy)],
  ]);
  $("ov-requests").textContent = s.requests.total;
  $("ov-errors").textContent = s.requests.errors;
  const p = s.requests.predict;
  $("ov-pred-count").textContent = p.count;
  $("ov-pred-latency").textContent = p.avg_ms == null ? "no predictions yet" : `${p.avg_ms} / ${p.p50_ms} / ${p.p95_ms} ms`;
  $("ov-memory").textContent = s.system.memory_mb + " MB";
  $("ov-cpu").textContent = `${s.system.cpu_percent}% of ${s.system.cpu_count} cores`;
  $("ov-pending").textContent = s.pending_uploads;
  $("ov-retrain").textContent = s.retraining.status === "running"
    ? `running (${s.retraining.progress}%)` : s.retraining.status;
}

function initOverview() {
  drawChart("chart-health", {
    type: "line",
    data: { labels: [], datasets: [{ label: "Response time (ms)", data: [], borderColor: "#2f5bd7",
      backgroundColor: "rgba(47,91,215,.10)", fill: true, tension: .3, pointRadius: 2, spanGaps: false }] },
    options: { plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "milliseconds" } }, x: { ticks: { maxTicksLimit: 8 } } } },
  });
  pingHealth(); loadStatus();
  setInterval(pingHealth, 5000);
  setInterval(loadStatus, 5000);
  setInterval(() => {
    $("ov-uptime").textContent = uptimeBase == null ? "–" : formatUptime(uptimeBase + (Date.now() - uptimeAt) / 1000);
  }, 1000);
}

/* -------------------------------------------------------------- predict */
let predictFile = null, lastPredictionId = null;

function setPredictFile(file, trueLabel = null) {
  predictFile = file; predictFile.trueLabel = trueLabel;
  $("predict-preview").src = URL.createObjectURL(file);
  $("predict-preview").hidden = false;
  $("dropzone-text").hidden = true;
  $("predict-filename").textContent = file.name + (trueLabel ? ` · true label: ${nice(trueLabel)}` : "");
  $("predict-btn").disabled = false;
}

async function runPrediction() {
  if (!predictFile) return;
  $("predict-btn").disabled = true; $("predict-btn").textContent = "Predicting…";
  $("predict-error").hidden = true;
  try {
    const form = new FormData();
    form.append("file", predictFile, predictFile.name);
    const started = performance.now();
    const r = await api("/predict", { method: "POST", body: form });
    const roundTrip = performance.now() - started;
    lastPredictionId = r.prediction_id;
    $("result-empty").hidden = true; $("result").hidden = false;
    $("res-label").textContent = nice(r.label);
    $("res-label").style.color = CLASS_COLORS[r.label] || "inherit";
    $("res-group").textContent = r.engagement;
    $("res-group").className = "badge " + (r.engagement === "Engaged" ? "engaged" : "not");
    $("res-confidence").textContent = `${pct(r.confidence)} confident`;
    $("res-engaged").textContent = pct(r.engaged_probability);
    $("res-ms").textContent = `${r.inference_ms} ms (round trip ${roundTrip.toFixed(0)} ms)`;
    $("res-version").textContent = `v${r.model_version} · ${r.served_by}`;
    const verdict = $("res-verdict");
    if (predictFile.trueLabel) {
      const ok = predictFile.trueLabel === r.label;
      verdict.hidden = false;
      verdict.className = "verdict " + (ok ? "ok" : "bad");
      verdict.innerHTML = ok ? `Correct<br><small>true label: ${esc(nice(predictFile.trueLabel))}</small>`
                             : `Incorrect<br><small>true label: ${esc(nice(predictFile.trueLabel))}</small>`;
      // Known label: log it as feedback so live accuracy is tracked.
      api("/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: r.prediction_id, true_label: predictFile.trueLabel }) }).catch(() => {});
    } else { verdict.hidden = true; }
    $("feedback-label").value = r.label; $("feedback-msg").textContent = "";
    const names = Object.keys(r.probabilities).sort((a, b) => r.probabilities[b] - r.probabilities[a]);
    drawChart("chart-proba", {
      type: "bar",
      data: { labels: names.map(nice), datasets: [{ data: names.map((n) => r.probabilities[n] * 100),
        backgroundColor: names.map((n) => CLASS_COLORS[n] || "#888"), borderRadius: 4 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => c.parsed.x.toFixed(1) + "%" } } },
        scales: { x: { min: 0, max: 100, title: { display: true, text: "probability (%)" } } } },
    });
  } catch (err) {
    $("predict-error").textContent = "Prediction failed: " + err.message; $("predict-error").hidden = false;
  } finally {
    $("predict-btn").disabled = false; $("predict-btn").textContent = "Predict";
  }
}

async function loadSamples() {
  if ($("samples").childElementCount) return;
  let list = [];
  try { list = await api("/samples?per_class=2"); } catch (_) { /* no test data on this instance */ }
  if (!list.length) { $("samples").innerHTML = '<span class="muted small">No test images on this instance.</span>'; return; }
  $("samples").innerHTML = list.map((s, i) =>
    `<button class="sample" data-i="${i}" title="Predict this ${esc(nice(s.label))} image"><img src="${esc(s.url)}" alt="${esc(nice(s.label))} sample" loading="lazy">${esc(nice(s.label))}</button>`).join("");
  $("samples").querySelectorAll(".sample").forEach((button) => button.addEventListener("click", async () => {
    const s = list[Number(button.dataset.i)];
    const blob = await (await fetch(s.url)).blob();
    setPredictFile(new File([blob], s.name, { type: blob.type || "image/jpeg" }), s.label);
    runPrediction();
  }));
}

function initPredict() {
  fillSelect($("feedback-label"));
  $("predict-file").addEventListener("change", (e) => { if (e.target.files[0]) setPredictFile(e.target.files[0]); });
  $("predict-btn").addEventListener("click", runPrediction);
  const zone = $("dropzone");
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setPredictFile(f); });
  $("feedback-btn").addEventListener("click", async () => {
    if (!lastPredictionId) return;
    try {
      await api("/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: lastPredictionId, true_label: $("feedback-label").value }) });
      $("feedback-msg").textContent = "Saved. It now counts towards live accuracy.";
    } catch (err) { $("feedback-msg").textContent = err.message; }
  });
  loaders.predict = loadSamples;
}

/* ------------------------------------------------------------- insights */
let insightsLoaded = false;

async function loadInsights(force = false) {
  if (insightsLoaded && !force) return;
  let d;
  try { d = await api("/insights"); } catch (err) {
    $("feature-cards").innerHTML = `<div class="alert error">Could not load insights: ${esc(err.message)}</div>`; return;
  }
  insightsLoaded = true;
  const names = d.class_names;
  drawChart("chart-classes", {
    type: "bar",
    data: { labels: names.map(nice), datasets: [
      { label: "Train", data: names.map((n) => d.counts.train[n]), backgroundColor: names.map((n) => CLASS_COLORS[n]), borderRadius: 4 },
      { label: "Test", data: names.map((n) => d.counts.test[n]), backgroundColor: names.map((n) => CLASS_COLORS[n] + "66"), borderRadius: 4 },
    ] },
    options: { plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
      scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: "images (solid: train, light: test)" } } } },
  });
  $("story-balance").textContent = d.balance_story;
  $("mean-images").innerHTML = names.filter((n) => d.mean_images[n]).map((n) =>
    `<figure><img src="/insights/mean/${esc(d.mean_images[n])}?t=${encodeURIComponent(d.generated_at)}" alt="Average ${esc(nice(n))} frame"><figcaption>${esc(nice(n))}</figcaption></figure>`).join("");
  $("story-mean").textContent = "Every class averages to a centred head-and-shoulders silhouette, so the classes are " +
    "not separated by where the student sits. The model has to read fine facial detail (eyes, mouth, head tilt), " +
    "which is why a pretrained convolutional backbone is used instead of simple pixel statistics.";

  const host = $("feature-cards");
  host.innerHTML = "";
  const keys = Object.keys(d.features);
  for (let i = 0; i < keys.length; i += 2) {
    const row = document.createElement("div"); row.className = "grid two";
    for (const key of keys.slice(i, i + 2)) {
      const f = d.features[key];
      const card = document.createElement("div"); card.className = "card";
      card.innerHTML = `<div class="feature-head"><h2>${esc(f.label)}</h2><span class="unit">${esc(f.unit)}</span></div>
        <p class="muted">${esc(f.meaning)}</p>
        <div class="chart-box"><canvas id="chart-feat-${key}"></canvas></div>
        <p class="story">${esc(f.story)}${f.interpretation ? " " + esc(f.interpretation) : ""}</p>`;
      row.appendChild(card);
    }
    host.appendChild(row);
    for (const key of keys.slice(i, i + 2)) {
      const f = d.features[key];
      const present = names.filter((n) => f.per_class[n]);
      const pad = (f.range[1] - f.range[0]) * 0.08;      // axis spans the 1st-99th percentile
      drawChart("chart-feat-" + key, {
        data: { labels: present.map(nice), datasets: [
          { type: "line", label: "Median", data: present.map((n) => f.per_class[n].median), showLine: false,
            pointStyle: "line", pointRadius: 22, pointHoverRadius: 22, pointBorderWidth: 3, borderColor: "#171a22", order: 0 },
          { type: "bar", label: "Middle 50% of images", data: present.map((n) => [f.per_class[n].q1, f.per_class[n].q3]),
            backgroundColor: present.map((n) => CLASS_COLORS[n] + "cc"), borderRadius: 4, borderSkipped: false, order: 1 },
        ] },
        options: { plugins: { legend: { labels: { usePointStyle: true } }, tooltip: { callbacks: { label: (c) => {
          const s = f.per_class[present[c.dataIndex]];
          return c.dataset.type === "bar" ? `25%–75%: ${s.q1.toFixed(1)} to ${s.q3.toFixed(1)} (mean ${s.mean.toFixed(1)})` : `median: ${s.median.toFixed(1)}`;
        } } } }, scales: { y: { beginAtZero: false, min: Math.floor(f.range[0] - pad), max: Math.ceil(f.range[1] + pad),
          title: { display: true, text: f.unit } } } },
      });
    }
  }
}

/* -------------------------------------------------------------- retrain */
let retrainTimer = null, lastRetrainStatus = null;

async function loadUploads() {
  let u;
  try { u = await api("/uploads"); } catch (_) { return; }
  $("uploads-table").querySelector("tbody").innerHTML = CLASSES.map((c) => {
    const row = u.by_label[c] || { pending: 0, processed: 0 };
    return `<tr><td>${nice(c)}</td><td class="num">${row.pending}</td><td class="num">${row.processed}</td></tr>`;
  }).join("") + `<tr><td><strong>Total</strong></td><td class="num"><strong>${u.totals.pending}</strong></td><td class="num"><strong>${u.totals.processed}</strong></td></tr>`;
  $("retrain-hint").textContent = u.totals.pending ? `${u.totals.pending} new image(s) waiting.` : "No new uploads waiting.";
}

function renderCompare(result) {
  const table = $("retrain-compare");
  if (!result || !result.before || !result.after) { table.hidden = true; return; }
  const rows = [["Accuracy", "accuracy", true], ["Loss", "loss", false], ["Precision (macro)", "precision_macro", true],
    ["Recall (macro)", "recall_macro", true], ["F1 (macro)", "f1_macro", true], ["ROC-AUC", "roc_auc_ovr", true]];
  table.querySelector("tbody").innerHTML = rows.map(([label, key, higherIsBetter]) => {
    const b = result.before[key], a = result.after[key];
    if (b == null || a == null) return "";
    const delta = a - b, better = higherIsBetter ? delta > 0 : delta < 0;
    const cls = Math.abs(delta) < 1e-9 ? "" : better ? "up" : "down";
    return `<tr><td>${label}</td><td class="num">${b.toFixed(4)}</td><td class="num">${a.toFixed(4)}</td><td class="num ${cls}">${delta >= 0 ? "+" : ""}${delta.toFixed(4)}</td></tr>`;
  }).join("");
  table.hidden = false;
}

async function pollRetrain() {
  let s;
  try { s = await api("/retrain/status", { cache: "no-store" }); } catch (_) { return; }
  const running = s.status === "running";
  $("retrain-btn").disabled = running;
  $("retrain-btn").textContent = running ? "Retraining…" : "Start retraining";
  if (s.status !== "idle") {
    $("retrain-progress-wrap").hidden = false;
    $("retrain-stage").textContent = s.stage || s.status;
    $("retrain-pct").textContent = s.progress + "%";
    $("retrain-bar").style.width = s.progress + "%";
    const log = $("retrain-log"); log.textContent = (s.log || []).join("\n"); log.scrollTop = log.scrollHeight;
  }
  const box = $("retrain-result");
  if (s.status === "completed" && s.result) {
    box.hidden = false; box.className = "alert " + (s.result.promoted ? "ok" : "warn");
    box.textContent = s.result.summary; renderCompare(s.result);
  } else if (s.status === "failed") {
    box.hidden = false; box.className = "alert error"; box.textContent = s.message; $("retrain-compare").hidden = true;
  } else if (running) { box.hidden = true; $("retrain-compare").hidden = true; }

  const t = s.triggers;
  $("trigger-table").innerHTML = `
    <tr><th>Automatic retraining</th><td>${t.auto_enabled ? "enabled" : "disabled"}</td></tr>
    <tr><th>Volume trigger: new images waiting</th><td>${t.pending_images} / ${t.volume_threshold} ${t.volume_trigger ? '<span class="pill warn">fires</span>' : ""}</td></tr>
    <tr><th>Drift trigger: mean confidence, last ${t.drift_window} predictions</th><td>${t.mean_recent_confidence == null ? "no predictions yet" : pct(t.mean_recent_confidence)} (floor ${pct(t.drift_confidence_floor, 0)}, ${t.drift_window_filled}/${t.drift_window} logged) ${t.drift_detected ? '<span class="pill warn">drift</span>' : ""}</td></tr>
    <tr><th>Recommendation</th><td>${t.retraining_recommended ? '<span class="pill warn">retrain recommended</span>' : '<span class="pill ok">model healthy</span>'}</td></tr>`;

  if (running) loadUploads();
  if (lastRetrainStatus === "running" && !running) { loadUploads(); loadHistory(); loadStatus(); insightsLoaded = false; }
  lastRetrainStatus = s.status;
  clearTimeout(retrainTimer);
  if ($("view-retrain").classList.contains("active") || running) retrainTimer = setTimeout(pollRetrain, running ? 1500 : 6000);
}

async function loadHistory() {
  let runs = [];
  try { runs = await api("/retrain/history"); } catch (_) { return; }
  const body = $("history-table").querySelector("tbody");
  if (!runs.length) { body.innerHTML = '<tr><td colspan="10" class="muted">No retraining runs yet.</td></tr>'; return; }
  body.innerHTML = runs.map((r) => {
    const statusClass = r.status === "completed" ? "ok" : r.status === "failed" ? "bad" : "info";
    const outcome = r.status !== "completed" ? esc(r.message || "") : r.promoted
      ? `<span class="pill ok">promoted to v${r.model_version}</span>` : '<span class="pill warn">kept previous model</span>';
    return `<tr><td>#${r.id}</td><td>${esc((r.started_at || "").replace("T", " ").replace("+00:00", ""))}</td><td>${esc(r.trigger_type)}</td>
      <td><span class="pill ${statusClass}">${esc(r.status)}</span></td><td class="num">${r.new_images ?? 0}</td><td class="num">${r.train_images ?? 0}</td>
      <td class="num">${r.epochs ?? 0}</td><td class="num">${r.metrics_before ? pct(r.metrics_before.accuracy) : "–"}</td>
      <td class="num">${r.metrics_after ? pct(r.metrics_after.accuracy) : "–"}</td><td>${outcome}</td></tr>`;
  }).join("");
}

function initRetrain() {
  fillSelect($("upload-label"));
  $("upload-files").addEventListener("change", (e) => {
    const n = e.target.files.length;
    $("upload-btn").disabled = !n;
    $("upload-count").textContent = n ? `${n} file(s) selected` : "";
  });
  $("upload-btn").addEventListener("click", async () => {
    const files = $("upload-files").files;
    if (!files.length) return;
    const form = new FormData();
    form.append("label", $("upload-label").value);
    for (const f of files) form.append("files", f, f.name);
    const box = $("upload-result");
    $("upload-btn").disabled = true; $("upload-btn").textContent = "Uploading…";
    try {
      const r = await api("/upload", { method: "POST", body: form });
      box.hidden = false; box.className = "alert " + (r.saved ? "ok" : "warn");
      let text = `Saved ${r.saved} image(s) to the database`;
      if (r.duplicates) text += `, skipped ${r.duplicates} duplicate(s)`;
      if (r.rejected.length) text += `, rejected ${r.rejected.length} (${r.rejected.slice(0, 3).map((x) => x.file + ": " + x.reason).join("; ")})`;
      text += `. ${r.pending_total} image(s) are now waiting for retraining.`;
      if (r.auto_retrain_started) text += " The volume threshold was reached, so retraining started automatically.";
      box.textContent = text;
      $("upload-files").value = ""; $("upload-count").textContent = "";
    } catch (err) {
      box.hidden = false; box.className = "alert error"; box.textContent = "Upload failed: " + err.message;
    } finally {
      $("upload-btn").textContent = "Upload to database"; $("upload-btn").disabled = !$("upload-files").files.length;
      loadUploads(); pollRetrain();
    }
  });
  $("retrain-btn").addEventListener("click", async () => {
    const box = $("retrain-result");
    try {
      await api("/retrain", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: Number($("retrain-epochs").value) || null, force: $("retrain-force").checked }) });
      box.hidden = true;
    } catch (err) {
      box.hidden = false; box.className = "alert warn"; box.textContent = err.message;
    }
    pollRetrain();
  });
  loaders.retrain = () => { loadUploads(); loadHistory(); pollRetrain(); };
}

/* ----------------------------------------------------------- evaluation */
function renderEvaluation(e) {
  $("eval-meta").textContent = `Model v${e.model_version} · ${e.n_samples} test images · ` +
    (e.from_training ? `metrics stored at training time (${(e.evaluated_at || "").replace("T", " ").replace("+00:00", " UTC")})`
                     : `evaluated live ${(e.evaluated_at || "").replace("T", " ").replace("+00:00", " UTC")} in ${e.duration_s}s`);
  metricTiles($("eval-metrics"), [
    ["Accuracy", pct(e.accuracy)], ["Loss", num(e.loss)], ["Precision", pct(e.precision_macro)],
    ["Recall", pct(e.recall_macro)], ["F1 score", pct(e.f1_macro)], ["ROC-AUC", num(e.roc_auc_ovr)],
    ["Engaged vs not", pct(e.engaged_vs_not_accuracy)],
  ]);
  const names = e.class_names, cm = e.confusion_matrix;
  let html = "<tr><th></th>" + names.map((n) => `<th>${esc(nice(n))}</th>`).join("") + "</tr>";
  cm.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<tr><th>${esc(nice(names[i]))}</th>` + row.map((v, j) => {
      const share = v / total;
      const bg = i === j ? `rgba(23,132,91,${0.12 + share * 0.78})` : v ? `rgba(192,57,47,${0.12 + share * 0.78})` : "#f4f5f8";
      return `<td style="background:${bg};color:${share > 0.55 ? "#fff" : "#171a22"}" title="true ${esc(nice(names[i]))}, predicted ${esc(nice(names[j]))}">${v}</td>`;
    }).join("") + "</tr>";
  });
  $("eval-matrix").innerHTML = html;
  drawChart("chart-perclass", {
    type: "bar",
    data: { labels: names.map(nice), datasets: [
      { label: "Precision", data: names.map((n) => e.per_class[n].precision * 100), backgroundColor: "#2f5bd7", borderRadius: 3 },
      { label: "Recall", data: names.map((n) => e.per_class[n].recall * 100), backgroundColor: "#1b9e77", borderRadius: 3 },
      { label: "F1", data: names.map((n) => e.per_class[n].f1 * 100), backgroundColor: "#d98b1f", borderRadius: 3 },
    ] },
    options: { scales: { y: { min: 0, max: 100, title: { display: true, text: "%" } } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` } } } },
  });
}

async function loadMonitoring() {
  let m;
  try { m = await api("/monitoring", { cache: "no-store" }); } catch (_) { return; }
  $("mon-total").textContent = m.total;
  $("mon-conf").textContent = m.mean_confidence == null ? "–" : pct(m.mean_confidence);
  $("mon-fb").textContent = m.feedback_count;
  $("mon-fbacc").textContent = m.feedback_accuracy == null ? "–" : pct(m.feedback_accuracy);
  const series = m.confidence_series.map((v) => v * 100);
  drawChart("chart-confidence", {
    type: "line",
    data: { labels: series.map((_, i) => i + 1), datasets: [
      { label: "Confidence", data: series, borderColor: "#2f5bd7", backgroundColor: "rgba(47,91,215,.1)", fill: true, tension: .25, pointRadius: 2 },
      { label: "Drift floor", data: series.map(() => m.triggers.drift_confidence_floor * 100), borderColor: "#c0392f", borderDash: [6, 5], pointRadius: 0, borderWidth: 1.5 },
    ] },
    options: { scales: { y: { min: 0, max: 100, title: { display: true, text: "confidence (%)" } },
      x: { title: { display: true, text: "most recent predictions, oldest to newest" }, ticks: { maxTicksLimit: 10 } } } },
  });
  const names = CLASSES.filter((c) => m.distribution[c]);
  drawChart("chart-dist", {
    type: "doughnut",
    data: { labels: names.map(nice), datasets: [{ data: names.map((n) => m.distribution[n]), backgroundColor: names.map((n) => CLASS_COLORS[n]) }] },
    options: { plugins: { legend: { position: "right" }, title: { display: true, text: "Predicted classes so far" } } },
  });
  $("recent-table").querySelector("tbody").innerHTML = m.recent.length ? m.recent.map((r) => {
    const fb = r.true_label == null ? "–" : r.true_label === r.predicted
      ? '<span class="pill ok">correct</span>' : `<span class="pill bad">was ${esc(nice(r.true_label))}</span>`;
    return `<tr><td>${esc(r.created_at.replace("T", " ").replace("+00:00", ""))}</td><td>${esc(r.filename || "")}</td><td>${esc(nice(r.predicted))}</td>
      <td class="num">${pct(r.confidence)}</td><td class="num">${r.latency_ms == null ? "–" : r.latency_ms.toFixed(1) + " ms"}</td><td>${fb}</td></tr>`;
  }).join("") : '<tr><td colspan="6" class="muted">No predictions logged yet.</td></tr>';
}

function initEvaluation() {
  $("eval-btn").addEventListener("click", async () => {
    $("eval-btn").disabled = true; $("eval-btn").textContent = "Evaluating…";
    try { renderEvaluation(await api("/evaluate", { method: "POST" })); }
    catch (err) { $("eval-meta").textContent = "Evaluation failed: " + err.message; }
    finally { $("eval-btn").disabled = false; $("eval-btn").textContent = "Run evaluation now"; }
  });
  loaders.evaluation = async () => {
    try { renderEvaluation(await api("/evaluate")); } catch (err) { $("eval-meta").textContent = err.message; }
    loadMonitoring();
  };
}

/* ----------------------------------------------------------------- boot */
loaders.insights = () => loadInsights();
initOverview();
initPredict();
initRetrain();
initEvaluation();
