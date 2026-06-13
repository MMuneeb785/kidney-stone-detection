import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "kidney_stone.onnx"
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']   # exact training order

# Class display config: label, color key
CLASS_CONFIG = {
    'Cyst':   {'icon': 'warn',  'color': 'warn'},
    'Normal': {'icon': 'safe',  'color': 'safe'},
    'Stone':  {'icon': 'alert', 'color': 'danger'},
    'Tumor':  {'icon': 'alert', 'color': 'danger'},
}

# ── Load ONNX model ────────────────────────────────────────────────────────────
session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
print("✅  ONNX model loaded — 4 classes:", CLASS_NAMES)

# ── Preprocessing ──────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, :]

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Kidney Diagnosis AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

  :root{
    --bg:#070b14;--surface:#0d1424;--border:#1a2540;
    --accent:#00e5ff;--accent2:#7b5cfa;
    --danger:#ff4f6b;--warn:#ffb347;--safe:#00e096;--info:#7b5cfa;
    --text:#e8edf8;--muted:#5a6a8a;
  }

  body{
    background:var(--bg);color:var(--text);
    font-family:'DM Sans',sans-serif;
    min-height:100vh;display:flex;flex-direction:column;
    align-items:center;padding:48px 20px 80px;overflow-x:hidden;
  }

  body::before{
    content:'';position:fixed;inset:0;
    background-image:
      linear-gradient(rgba(0,229,255,.03) 1px,transparent 1px),
      linear-gradient(90deg,rgba(0,229,255,.03) 1px,transparent 1px);
    background-size:48px 48px;pointer-events:none;z-index:0;
  }

  .blob{position:fixed;border-radius:50%;filter:blur(120px);opacity:.3;pointer-events:none;z-index:0;}
  .blob1{width:500px;height:500px;background:#00e5ff;top:-160px;left:-160px;}
  .blob2{width:400px;height:400px;background:#7b5cfa;bottom:-120px;right:-120px;}

  .wrapper{position:relative;z-index:1;width:100%;max-width:660px;}

  /* Header */
  header{text-align:center;margin-bottom:44px;}
  .logo-ring{
    width:72px;height:72px;border-radius:50%;
    border:2px solid var(--accent);
    display:flex;align-items:center;justify-content:center;
    margin:0 auto 18px;
    animation:pulse 3s ease-in-out infinite;
  }
  .logo-ring svg{width:34px;height:34px;color:var(--accent);}
  @keyframes pulse{
    0%,100%{box-shadow:0 0 20px rgba(0,229,255,.2);}
    50%{box-shadow:0 0 55px rgba(0,229,255,.55);}
  }
  h1{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(1.8rem,5vw,2.5rem);letter-spacing:-1px;line-height:1.1;}
  h1 span{color:var(--accent);}
  .subtitle{margin-top:10px;color:var(--muted);font-size:.9rem;}
  .badge-row{display:flex;justify-content:center;gap:8px;margin-top:14px;flex-wrap:wrap;}
  .chip{padding:4px 12px;border-radius:100px;font-size:.72rem;font-weight:500;border:1px solid var(--border);color:var(--muted);}

  /* Card */
  .card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:32px;}

  /* Drop zone */
  .drop-zone{
    border:2px dashed var(--border);border-radius:14px;
    padding:44px 24px;text-align:center;cursor:pointer;
    transition:border-color .25s,background .25s;
    position:relative;overflow:hidden;
  }
  .drop-zone:hover,.drop-zone.dragging{border-color:var(--accent);background:rgba(0,229,255,.04);}
  .drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
  .drop-icon{
    width:52px;height:52px;margin:0 auto 14px;
    background:linear-gradient(135deg,#0d1f38,#1a2f50);
    border-radius:14px;display:flex;align-items:center;justify-content:center;
    border:1px solid var(--border);
  }
  .drop-icon svg{width:24px;height:24px;color:var(--accent);}
  .drop-title{font-family:'Syne',sans-serif;font-weight:600;font-size:1rem;}
  .drop-sub{color:var(--muted);font-size:.82rem;margin-top:6px;}

  /* Preview */
  #preview-wrap{display:none;margin-top:18px;border-radius:12px;overflow:hidden;border:1px solid var(--border);position:relative;}
  #preview-wrap img{width:100%;max-height:280px;object-fit:cover;display:block;}
  .preview-label{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(7,11,20,.9));padding:14px 12px 8px;font-size:.78rem;color:var(--muted);}

  /* Button */
  .btn{
    width:100%;margin-top:18px;padding:15px;border:none;border-radius:12px;
    background:linear-gradient(135deg,var(--accent2),var(--accent));
    color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;
    cursor:pointer;transition:opacity .2s,transform .15s;
  }
  .btn:hover{opacity:.9;transform:translateY(-1px);}
  .btn:disabled{opacity:.4;cursor:not-allowed;transform:none;}
  .spinner{display:none;width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;margin:0 auto;}
  @keyframes spin{to{transform:rotate(360deg);}}

  /* Result */
  #result{display:none;margin-top:26px;}
  .divider{height:1px;background:var(--border);margin:22px 0;}

  .verdict-card{
    border-radius:14px;padding:20px 22px;
    display:flex;align-items:center;gap:16px;margin-bottom:22px;
  }
  .verdict-card.danger{background:rgba(255,79,107,.08);border:1px solid rgba(255,79,107,.25);}
  .verdict-card.warn  {background:rgba(255,179,71,.08); border:1px solid rgba(255,179,71,.25);}
  .verdict-card.safe  {background:rgba(0,224,150,.08);  border:1px solid rgba(0,224,150,.25);}
  .verdict-card.info  {background:rgba(123,92,250,.08); border:1px solid rgba(123,92,250,.25);}

  .verdict-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
  .verdict-card.danger .verdict-icon{background:rgba(255,79,107,.15);}
  .verdict-card.warn   .verdict-icon{background:rgba(255,179,71,.15);}
  .verdict-card.safe   .verdict-icon{background:rgba(0,224,150,.15);}
  .verdict-card.info   .verdict-icon{background:rgba(123,92,250,.15);}

  .verdict-label{font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;}
  .verdict-card.danger .verdict-label{color:var(--danger);}
  .verdict-card.warn   .verdict-label{color:var(--warn);}
  .verdict-card.safe   .verdict-label{color:var(--safe);}
  .verdict-card.info   .verdict-label{color:var(--info);}
  .verdict-conf{font-size:.82rem;color:var(--muted);margin-top:3px;}

  /* Bars */
  .bar-row{margin-bottom:12px;}
  .bar-meta{display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:5px;}
  .bar-label{color:var(--text);font-weight:500;}
  .bar-pct{color:var(--muted);}
  .bar-track{height:7px;border-radius:100px;background:var(--border);overflow:hidden;}
  .bar-fill{height:100%;border-radius:100px;width:0;transition:width 1s cubic-bezier(.16,1,.3,1);}
  .bar-fill.danger{background:linear-gradient(90deg,#ff4f6b,#ff8a9a);}
  .bar-fill.warn  {background:linear-gradient(90deg,#ffb347,#ffd088);}
  .bar-fill.safe  {background:linear-gradient(90deg,#00c87e,#00e096);}
  .bar-fill.info  {background:linear-gradient(90deg,#7b5cfa,#a48dfc);}

  .error-box{display:none;margin-top:14px;padding:13px 15px;border-radius:10px;background:rgba(255,79,107,.08);border:1px solid rgba(255,79,107,.25);color:var(--danger);font-size:.85rem;}
  .note{margin-top:26px;text-align:center;font-size:.75rem;color:var(--muted);line-height:1.7;}
</style>
</head>
<body>
<div class="blob blob1"></div>
<div class="blob blob2"></div>

<div class="wrapper">
  <header>
    <div class="logo-ring">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <h1>Kidney <span>Diagnosis</span> AI</h1>
    <p class="subtitle">ResNet-18 · ONNX Runtime · Medical Imaging</p>
    <div class="badge-row">
      <span class="chip">🔴 Cyst</span>
      <span class="chip">🟢 Normal</span>
      <span class="chip">🟠 Stone</span>
      <span class="chip">🔴 Tumor</span>
    </div>
  </header>

  <div class="card">
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/*"/>
      <div class="drop-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div class="drop-title">Drop CT / X-ray image here</div>
      <div class="drop-sub">or click to browse &nbsp;·&nbsp; JPG, PNG, WEBP</div>
    </div>

    <div id="preview-wrap">
      <img id="preview" src="" alt="Preview"/>
      <div class="preview-label" id="fileName"></div>
    </div>

    <div class="error-box" id="errorBox"></div>

    <button class="btn" id="analyzeBtn" disabled onclick="analyze()">
      <span id="btnText">Select an image first</span>
      <div class="spinner" id="spinner"></div>
    </button>

    <div id="result">
      <div class="divider"></div>
      <div class="verdict-card" id="verdictCard">
        <div class="verdict-icon" id="verdictIcon"></div>
        <div>
          <div class="verdict-label" id="verdictLabel"></div>
          <div class="verdict-conf" id="verdictConf"></div>
        </div>
      </div>
      <div id="barsContainer"></div>
    </div>
  </div>

  <p class="note">
    For research &amp; educational use only.<br/>
    Always consult a qualified medical professional for clinical diagnosis.
  </p>
</div>

<script>
const CLASS_COLOR = {
  'Cyst':   'info',
  'Normal': 'safe',
  'Stone':  'warn',
  'Tumor':  'danger'
};

const CLASS_SVG = {
  danger: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--danger)"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
  warn:   `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--warn)"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
  safe:   `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--safe)"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
  info:   `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--info)"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`
};

const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');
const previewImg  = document.getElementById('preview');
const previewWrap = document.getElementById('preview-wrap');
const fileNameEl  = document.getElementById('fileName');
const btn         = document.getElementById('analyzeBtn');
const btnText     = document.getElementById('btnText');
const spinner     = document.getElementById('spinner');
const resultEl    = document.getElementById('result');
const errorBox    = document.getElementById('errorBox');
let selectedFile  = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragging');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => { previewImg.src = e.target.result; previewWrap.style.display = 'block'; fileNameEl.textContent = file.name; };
  reader.readAsDataURL(file);
  btn.disabled = false;
  btnText.textContent = 'Analyze Image';
  resultEl.style.display = 'none';
  errorBox.style.display = 'none';
}

async function analyze() {
  if (!selectedFile) return;
  btn.disabled = true; btnText.style.display = 'none'; spinner.style.display = 'block';
  errorBox.style.display = 'none'; resultEl.style.display = 'none';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    errorBox.textContent = '⚠  ' + (err.message || 'Something went wrong.');
    errorBox.style.display = 'block';
  } finally {
    btn.disabled = false; btnText.style.display = 'block'; spinner.style.display = 'none';
    btnText.textContent = 'Analyze Again';
  }
}

function showResult(data) {
  const color = CLASS_COLOR[data.prediction] || 'info';
  const card  = document.getElementById('verdictCard');
  card.className = 'verdict-card ' + color;
  document.getElementById('verdictIcon').innerHTML  = CLASS_SVG[color];
  document.getElementById('verdictLabel').textContent = data.prediction + ' Detected';
  document.getElementById('verdictConf').textContent  = 'Confidence: ' + data.confidence;

  const bars = document.getElementById('barsContainer');
  bars.innerHTML = '';
  for (const [cls, pct] of Object.entries(data.probabilities)) {
    const val = parseFloat(pct);
    const c   = CLASS_COLOR[cls] || 'info';
    bars.innerHTML += `
      <div class="bar-row">
        <div class="bar-meta"><span class="bar-label">${cls}</span><span class="bar-pct">${pct}</span></div>
        <div class="bar-track"><div class="bar-fill ${c}" data-val="${val}" style="width:0%"></div></div>
      </div>`;
  }

  resultEl.style.display = 'block';
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill[data-val]').forEach(el => { el.style.width = el.dataset.val + '%'; });
  });
}
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Send an image file with key 'image'"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = preprocess(img)
    logits = session.run(None, {INPUT_NAME: tensor})[0][0]
    probs  = softmax(logits)
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": f"{round(float(probs[pred_idx]) * 100, 2)}%",
        "probabilities": {
            CLASS_NAMES[i]: f"{round(float(probs[i]) * 100, 2)}%"
            for i in range(len(CLASS_NAMES))
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)