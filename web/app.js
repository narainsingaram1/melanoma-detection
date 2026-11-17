/* In-browser inference for the melanoma demo.
 * Mirrors app.py / src/inference.py exactly:
 *   decode -> PIL-equivalent bilinear resize to 224x224 -> ImageNet normalize
 *   -> ONNX graph (calibrated probs, Mahalanobis OOD score, per-class Grad-CAM at 7x7)
 *   -> ReLU, OpenCV-equivalent bilinear upsample to 224, min-max normalize, inferno colormap.
 * The ONNX graph is built and checked against the PyTorch app by scripts/build_web.py. */
'use strict';

const ORT_VERSION = '1.22.0';
const SIZE = 224;
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
const CLASS_NAMES = {
  akiec: 'Actinic Keratoses', bcc: 'Basal Cell Carcinoma', bkl: 'Benign Keratosis',
  df: 'Dermatofibroma', mel: 'Melanoma', nv: 'Melanocytic Nevi', vasc: 'Vascular Lesions',
};
const RISK = { mel: 'malignant', bcc: 'malignant', akiec: 'pre-malignant', bkl: 'benign', df: 'benign', nv: 'benign', vasc: 'benign' };
const RISK_LABEL = { malignant: 'Malignant class', 'pre-malignant': 'Pre-malignant class', benign: 'Benign class' };
const SAMPLES = ['akiec_ISIC_0024511', 'bcc_ISIC_0024345', 'bkl_ISIC_0024336', 'df_ISIC_0024994',
  'mel_ISIC_0024367', 'nv_ISIC_0024306', 'vasc_ISIC_0024669'];

const $ = (id) => document.getElementById(id);
const state = { session: null, meta: null, lut: null, bitmap: null, busy: false };

/* ---------- Tabs ---------- */
document.querySelectorAll('.tabs button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
    document.querySelectorAll('.panel').forEach((p) => { p.hidden = p.id !== 'tab-' + btn.dataset.tab; });
  });
});

/* ---------- Static content ---------- */
$('classRef').innerHTML = ['mel', 'bcc', 'akiec', 'bkl', 'df', 'nv', 'vasc'].map((c) =>
  `<li><span class="mx-dot risk-${RISK[c]}"></span><span class="mx-cls">${CLASS_NAMES[c]}</span>` +
  `<span class="mx-code">${c}</span><span class="mx-risk">${RISK_LABEL[RISK[c]].replace(' class', '')}</span></li>`).join('');

$('samples').innerHTML = SAMPLES.map((s) =>
  `<button type="button" data-src="samples/${s}.jpg" title="${CLASS_NAMES[s.split('_')[0]]} (test image)">` +
  `<img src="samples/${s}.jpg" alt="${CLASS_NAMES[s.split('_')[0]]} sample" loading="lazy"></button>`).join('');

fetch('data/results.html').then((r) => r.text()).then((html) => { $('resultsBody').innerHTML = html; })
  .catch(() => { $('resultsBody').textContent = 'Could not load results.'; });

/* ---------- Model loading ---------- */
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(Math.min(got / total, 1));
  }
  const out = new Uint8Array(got); let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function loadModel() {
  const btn = $('analyze'), bar = $('progressBar'), prog = $('progress');
  prog.hidden = false;
  try {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime failed to load (check your connection or ad blocker)');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const [meta, lut, bytes] = await Promise.all([
      fetch('data/model.json').then((r) => r.json()),
      fetch('data/inferno.json').then((r) => r.json()),
      fetchWithProgress('model/lesion.onnx', (f) => {
        bar.style.width = `${Math.round(f * 100)}%`;
        btn.textContent = `Loading model… ${Math.round(f * 100)}%`;
      }),
    ]);
    btn.textContent = 'Preparing model…';
    state.session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    state.meta = meta; state.lut = lut;
    $('meta').textContent = `EfficientNet-B0 · ${meta.variant} · seed ${meta.seed} · runs in your browser (ONNX Runtime Web)`;
    prog.hidden = true;
    btn.textContent = 'Analyze image';
    btn.disabled = !state.bitmap;
    if (state.bitmap) analyze();
  } catch (err) {
    console.error(err);
    prog.hidden = true;
    btn.textContent = 'Model failed to load';
    showNote('The model could not be loaded',
      `${escapeHtml(err.message || String(err))}. Reload the page to try again. A recent desktop or mobile browser is required.`, true);
  }
}

/* ---------- Image input ---------- */
async function setImageFromBlob(blob, { auto = false } = {}) {
  if (!blob || !blob.type.startsWith('image/')) return showNote('Unsupported file', 'Choose a JPEG, PNG or WebP image.', true);
  if (blob.size > MAX_BYTES) return showNote('Image too large', 'Upload an image under 10 MB.', true);
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  } catch (e) {
    return showNote('Could not read this image', 'The file may be corrupt or in a format this browser cannot decode.', true);
  }
  if (bitmap.width * bitmap.height > MAX_PIXELS) return showNote('Image too large', 'Upload a photo under 40 megapixels.', true);
  if (state.bitmap) state.bitmap.close();
  state.bitmap = bitmap;
  const prev = $('preview');
  if (prev.src.startsWith('blob:')) URL.revokeObjectURL(prev.src);
  prev.src = URL.createObjectURL(blob);
  prev.hidden = false; $('dropEmpty').hidden = true;
  $('analyze').disabled = !state.session;
  if (auto && state.session) analyze();
  else if (!state.session) showNote('Loading the model', 'The image will be analyzed as soon as the model has loaded.');
}

$('file').addEventListener('change', (e) => {
  clearActiveSample();
  setImageFromBlob(e.target.files[0], { auto: true });
  e.target.value = '';
});
const drop = $('drop');
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, () => drop.classList.remove('is-over')));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  clearActiveSample();
  setImageFromBlob(e.dataTransfer.files[0], { auto: true });
});
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) { clearActiveSample(); setImageFromBlob(item.getAsFile(), { auto: true }); }
});
$('samples').addEventListener('click', async (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  clearActiveSample(); b.classList.add('is-active');
  const blob = await fetch(b.dataset.src).then((r) => r.blob());
  setImageFromBlob(blob, { auto: true });
});
function clearActiveSample() { document.querySelectorAll('.samples button').forEach((x) => x.classList.remove('is-active')); }
$('analyze').addEventListener('click', () => analyze());

/* ---------- Preprocessing: port of Pillow's resample (BILINEAR, 8-bit fixed point) ---------- */
const PRECISION_BITS = 32 - 8 - 2;
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    const k = new Float64Array(ksize);
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k[x] = w; ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      // normalize_coeffs_8bpc
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PRECISION_BITS)) : Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}
const clip8 = (ss) => { const v = Math.floor(ss / (1 << PRECISION_BITS)); return v < 0 ? 0 : v > 255 ? 255 : v; };

function resizeRGB(src, w, h, ow, oh) {
  // src: RGBA Uint8ClampedArray. Horizontal pass then vertical pass, each rounded to 8 bits (as Pillow does).
  let cur = src, cw = w, stride = 4;
  if (ow !== w) {
    const { ksize, bounds, kk } = coeffs(w, ow);
    const out = new Uint8ClampedArray(ow * h * 3);
    for (let y = 0; y < h; y++) {
      for (let xx = 0; xx < ow; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
        for (let c = 0; c < 3; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let x = 0; x < xmax; x++) ss += cur[(y * cw + x + xmin) * stride + c] * kk[xx * ksize + x];
          out[(y * ow + xx) * 3 + c] = clip8(ss);
        }
      }
    }
    cur = out; cw = ow; stride = 3;
  }
  if (oh !== h) {
    const { ksize, bounds, kk } = coeffs(h, oh);
    const out = new Uint8ClampedArray(ow * oh * 3);
    for (let yy = 0; yy < oh; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
      for (let x = 0; x < ow; x++) {
        for (let c = 0; c < 3; c++) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let y = 0; y < ymax; y++) ss += cur[((y + ymin) * cw + x) * stride + c] * kk[yy * ksize + y];
          out[(yy * ow + x) * 3 + c] = clip8(ss);
        }
      }
    }
    cur = out; stride = 3;
  }
  if (stride === 4) { // no resize needed: drop alpha
    const out = new Uint8ClampedArray(ow * oh * 3);
    for (let i = 0, j = 0; i < out.length; i += 3, j += 4) { out[i] = cur[j]; out[i + 1] = cur[j + 1]; out[i + 2] = cur[j + 2]; }
    cur = out;
  }
  return cur;
}

function bitmapPixels(bitmap) {
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  // Opaque backdrop for transparent PNGs (PIL convert('RGB') drops alpha; lesion photos have none)
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height).data;
}

function toTensor(rgb, mean, std) {
  const n = SIZE * SIZE, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.fround(rgb[i * 3 + c] / 255);
      data[c * n + i] = Math.fround(Math.fround(v - mean[c]) / std[c]);
    }
  }
  return new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]);
}

/* ---------- Grad-CAM post-processing: ReLU -> cv2.resize INTER_LINEAR -> min-max ---------- */
function camTo224(cam7) {
  const relu = cam7.map((v) => (v > 0 ? v : 0));
  const out = new Float32Array(SIZE * SIZE), s = 7 / SIZE;
  for (let y = 0; y < SIZE; y++) {
    let fy = (y + 0.5) * s - 0.5; let y0 = Math.floor(fy); let dy = fy - y0;
    if (y0 < 0) { y0 = 0; dy = 0; } if (y0 >= 6) { y0 = 6; dy = 0; }
    const y1 = Math.min(y0 + 1, 6);
    for (let x = 0; x < SIZE; x++) {
      let fx = (x + 0.5) * s - 0.5; let x0 = Math.floor(fx); let dx = fx - x0;
      if (x0 < 0) { x0 = 0; dx = 0; } if (x0 >= 6) { x0 = 6; dx = 0; }
      const x1 = Math.min(x0 + 1, 6);
      out[y * SIZE + x] = (1 - dy) * ((1 - dx) * relu[y0 * 7 + x0] + dx * relu[y0 * 7 + x1]) +
        dy * ((1 - dx) * relu[y1 * 7 + x0] + dx * relu[y1 * 7 + x1]);
    }
  }
  let mx = -Infinity;
  for (const v of out) if (v > mx) mx = v;
  if (mx > 0) {
    let mn = Infinity; for (const v of out) if (v < mn) mn = v;
    for (let i = 0; i < out.length; i++) out[i] -= mn;
    let m2 = -Infinity; for (const v of out) if (v > m2) m2 = v;
    if (m2 > 0) for (let i = 0; i < out.length; i++) out[i] /= m2;
  }
  return out;
}

/* ---------- Inference ---------- */
async function runModel(bitmap) {
  const t0 = performance.now();
  const rgb = resizeRGB(bitmapPixels(bitmap), bitmap.width, bitmap.height, SIZE, SIZE);
  const feeds = { image: toTensor(rgb, state.meta.mean, state.meta.std) };
  const out = await state.session.run(feeds);
  const probs = Array.from(out.probs.data);
  const ood = out.ood.data[0];
  const top = probs.indexOf(Math.max(...probs));
  const cam7 = out.cam.data.slice(top * 49, top * 49 + 49);
  return { rgb, probs, ood, top, cam7, ms: performance.now() - t0 };
}

async function analyze() {
  if (!state.session || !state.bitmap || state.busy) return;
  state.busy = true;
  const btn = $('analyze');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  try {
    const bmp = state.bitmap;
    const r = await runModel(bmp);
    const m = state.meta, classes = m.classes;
    const tag = `${bmp.width}×${bmp.height} → 224×224 · T = ${m.temperature.toFixed(2)} · ` +
      `OOD ${Math.round(r.ood)} / ${Math.round(m.ood_threshold)} · ${Math.round(r.ms)} ms`;
    if (!(r.ood <= m.ood_threshold)) {
      showNote('Not a dermoscopy image',
        'This image is far from anything the model was trained on, so no prediction is shown. ' +
        `Upload a close-up dermoscopy image of a single skin lesion.<span class="mx-tag">${tag}</span>`);
      $('plate').innerHTML = '';
    } else {
      renderResult(classes, r, tag);
      renderPlate(r, classes[r.top]);
    }
    // Single-column layout: the result sits below the inputs, so bring it into view
    const res = $('result');
    if (matchMedia('(max-width: 900px)').matches && res.getBoundingClientRect().top > innerHeight * 0.6) {
      window.scrollTo({ top: window.scrollY + res.getBoundingClientRect().top - 16, behavior: 'auto' });
    }
  } catch (err) {
    console.error(err);
    showNote('Analysis failed', escapeHtml(err.message || String(err)), true);
  } finally {
    state.busy = false;
    btn.disabled = false; btn.textContent = 'Analyze image';
  }
}

/* ---------- Rendering ---------- */
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function showNote(title, body, isError = false) {
  $('result').innerHTML = `<div class="mx-card mx-note${isError ? ' is-error' : ''}"><div class="mx-note-title">${title}</div><p>${body}</p></div>`;
  $('plate').innerHTML = '';
}

function renderResult(classes, r, tag) {
  const ranked = classes.map((c, i) => [c, r.probs[i]]).sort((a, b) => b[1] - a[1]);
  const code = ranked[0][0];
  const rows = ranked.map(([c, p], i) =>
    `<li class="mx-row${i === 0 ? ' is-top' : ''}"><span class="mx-dot risk-${RISK[c]}"></span>` +
    `<span class="mx-cls">${CLASS_NAMES[c]}</span><span class="mx-code">${c}</span>` +
    `<span class="mx-track"><span class="mx-fill" style="width:${Math.max(p * 100, 0.5).toFixed(1)}%"></span></span>` +
    `<span class="mx-pct">${(p * 100).toFixed(1)}%</span></li>`).join('');
  const [[, p1], [c2, p2]] = ranked;
  const close = p1 - p2 < 0.10
    ? `<p class="mx-close">Close call: ${CLASS_NAMES[c2]} is ${((p1 - p2) * 100).toFixed(1)} points behind</p>` : '';
  $('result').innerHTML = `
    <div class="mx-card">
      <div class="mx-kicker">Most likely</div>
      <div class="mx-head"><div class="mx-title">${CLASS_NAMES[code]}</div><div class="mx-big">${(p1 * 100).toFixed(1)}%</div></div>
      <div class="mx-sub"><span class="mx-dot risk-${RISK[code]}"></span>${RISK_LABEL[RISK[code]]} · calibrated probability</div>${close}
      <ol class="mx-list">${rows}</ol>
      <div class="mx-tag">${tag}</div>
    </div>`;
}

function renderPlate(r, code) {
  const heat = camTo224(r.cam7), lut = state.lut, n = SIZE * SIZE;
  const inp = new Uint8ClampedArray(n * 4), hm = new Uint8ClampedArray(n * 4), ov = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const li = Math.trunc(255 * heat[i]) * 3;  // np.uint8(255 * heatmap) truncates
    for (let c = 0; c < 3; c++) {
      const a = r.rgb[i * 3 + c], h = lut[li + c];
      inp[i * 4 + c] = a; hm[i * 4 + c] = h;
      ov[i * 4 + c] = Math.trunc(Math.min(255, Math.max(0, 0.55 * a + 0.45 * h)));
    }
    inp[i * 4 + 3] = hm[i * 4 + 3] = ov[i * 4 + 3] = 255;
  }
  const cells = [['Input', inp], ['Grad-CAM', hm], [`Overlay · ${CLASS_NAMES[code]}`, ov]];
  $('plate').innerHTML = `
    <div class="mx-card">
      <div class="mx-kicker">Where the model looked</div>
      <div class="mx-plate">${cells.map(([t]) => `<figure><canvas width="${SIZE}" height="${SIZE}" aria-label="${t}"></canvas><figcaption>${t}</figcaption></figure>`).join('')}</div>
      <p class="mx-caption">Brighter regions contributed more to the predicted class. Grad-CAM on the last
        convolutional block of EfficientNet-B0, upsampled from 7×7</p>
    </div>`;
  $('plate').querySelectorAll('canvas').forEach((cv, i) => cv.getContext('2d').putImageData(new ImageData(cells[i][1], SIZE, SIZE), 0, 0));
}

/* Exposed for automated checks against the Python reference (scripts/build_web.py) */
window.__mx = {
  async check() {
    const rows = [];
    for (const s of SAMPLES) {
      const blob = await fetch(`samples/${s}.jpg`).then((x) => x.blob());
      const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
      const r = await runModel(bmp);
      const ref = state.meta.reference[`${s}.jpg`];
      const err = Math.max(...r.probs.map((p, i) => Math.abs(p - ref.probs[i])));
      rows.push({ s, pred: state.meta.classes[r.top], ref: ref.pred, maxProbErr: +err.toFixed(5), ood: Math.round(r.ood), refOod: Math.round(ref.ood), ms: Math.round(r.ms) });
    }
    return rows;
  },
};

loadModel();
