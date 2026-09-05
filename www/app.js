/* Capacitor / Play Store WebView shell — set API host once */
const API_BASE = (localStorage.getItem('apiBase') || '').replace(/\/$/, '') || '';
function api(path) {
  if (API_BASE) return `${API_BASE}${path}`;
  return path; // same-origin when served by FastAPI
}

let deferredPrompt = null;
const $ = (id) => document.getElementById(id);

async function loadScenes() {
  const res = await fetch(api('/api/scenes'));
  const data = await res.json();
  const sel = $('sceneSelect');
  sel.innerHTML = '';
  for (const s of data.scenes || []) {
    const opt = document.createElement('option');
    opt.value = s.scene_id;
    opt.textContent = `${s.scene_id} (${s.n_wrecks || 0} catalog)`;
    sel.appendChild(opt);
  }
}

function fmt(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

function showDetail(pin) {
  const card = $('detailCard');
  const body = $('detailBody');
  card.classList.remove('hidden');
  const alts = (pin.alternatives || []).map(a =>
    `<div class="alt"><b>${a.name}</b> · ${a.distance_m} m · score ${a.score}<br/><span style="color:#9bb0c5">${a.catalog_id}</span></div>`
  ).join('') || '<p class="status">No alternatives</p>';
  body.innerHTML = `
    <p><b>Likely vessel:</b> ${pin.best_guess || '—'}</p>
    <p><b>Ship confidence:</b> ${fmt(pin.ship_confidence)}</p>
    <p><b>Catalog ID:</b> ${pin.catalog_id || '—'}</p>
    <p><b>Rationale:</b> ${pin.rationale || '—'}</p>
    <p><b>History:</b> ${(pin.history || '').slice(0, 500) || '—'}</p>
    <h3 style="font-size:0.9rem;margin:12px 0 6px">Top alternatives</h3>
    ${alts}
  `;
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderPins(pins) {
  const tbody = $('pinsTable').querySelector('tbody');
  tbody.innerHTML = '';
  pins.forEach((p, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmt(p.lat)}</td>
      <td>${fmt(p.lon)}</td>
      <td>${fmt(p.detect_score)}</td>
      <td>${p.best_guess || '—'}</td>
      <td>${fmt(p.ship_confidence)}</td>
      <td>${p.catalog_id || '—'}</td>`;
    tr.addEventListener('click', () => {
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      showDetail(p);
    });
    tbody.appendChild(tr);
    if (idx === 0) tr.click();
  });
}

async function runDetect() {
  const sceneId = $('sceneSelect').value;
  const minConf = $('minConf').value;
  $('status').textContent = 'Running detection + catalog ship ID…';
  $('runBtn').disabled = true;
  try {
    const res = await fetch(api(`/api/detect?scene_id=${encodeURIComponent(sceneId)}&min_confidence=${minConf}`));
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $('overlayImg').src = `data:image/png;base64,${data.overlay_png_base64}`;
    const dm = data.detection_metrics || {};
    const im = data.identification_metrics || {};
    $('metrics').innerHTML = `
      <div class="metric"><span>Precision</span><b>${fmt(dm.precision)}</b></div>
      <div class="metric"><span>Recall</span><b>${fmt(dm.recall)}</b></div>
      <div class="metric"><span>F1</span><b>${fmt(dm.f1)}</b></div>
      <div class="metric"><span>ID P@1</span><b>${fmt(im.precision_at_1)}</b></div>`;
    renderPins(data.pins_table || []);
    $('status').textContent = `${data.n_catalog} catalog pins · ${data.n_detections} CV candidates · ${data.notes || ''}`;
  } catch (e) {
    $('status').textContent = `Error: ${e.message || e}`;
  } finally {
    $('runBtn').disabled = false;
  }
}

$('minConf').addEventListener('input', () => {
  $('minConfVal').textContent = Number($('minConf').value).toFixed(2);
});
$('runBtn').addEventListener('click', runDetect);

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  $('installBtn').classList.remove('hidden');
});
$('installBtn').addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  $('installBtn').classList.add('hidden');
});

if ('serviceWorker' in navigator) {
  const sw = API_BASE ? null : 'sw.js';
  if (sw) navigator.serviceWorker.register(sw).catch(() => {});
}

loadScenes().then(() => {
  $('status').textContent = 'Ready. Tap Detect & identify.';
}).catch((e) => {
  $('status').textContent = `Set API host (localStorage apiBase) or open via FastAPI. ${e}`;
});
