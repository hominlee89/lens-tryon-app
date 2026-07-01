import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video    = document.getElementById("video");
const canvas   = document.getElementById("overlay");
const ctx      = canvas.getContext("2d");
const beautyCanvas = document.getElementById("beauty");
video.style.display = "none";
const statusEl      = document.getElementById("status");
const productListEl = document.getElementById("product-list");
const captureBtn    = document.getElementById("capture-btn");
const topbar        = document.getElementById("topbar");

// 뷰티 엔진 (AR 이식): WebGL 셰이더 — 피부 스무딩·잡티/홍조 제거·얼굴 슬림·눈 확대
const softFocusEngine = new SoftFocusEngine(video, beautyCanvas);
const faceMaskCanvas = document.createElement("canvas");
const faceMaskCtx = faceMaskCanvas.getContext("2d");

// 비디오 픽셀 샘플러 (동공 검출·홍채 톤용, willReadFrequently로 빠른 read)
const sampler = document.createElement("canvas");
const sctx = sampler.getContext("2d", { willReadFrequently: true });

// ── 랜드마크 ────────────────────────────────────────────────────────────────
const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const LEFT_EYE_CONTOUR  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const RIGHT_EYE_CONTOUR = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382];
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
];

const HVID_MM = 11.7; // 홍채 가로 지름 평균 → mm↔px 환산 기준

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {};

// 뷰티 레벨 0~3 → 셰이더 강도
let beautyLevel = 2;
const BEAUTY_STRENGTH = [0, 0.5, 0.75, 0.95];

// ── 추적 상태 (스무딩 + 속도 예측) ──────────────────────────────────────────
const track = { left: null, right: null };
const SMOOTH  = 0.55;
const PREDICT = 0.5;

function trackGeo(side, raw) {
  const st = track[side];
  if (!st) { track[side] = { ...raw, vx: 0, vy: 0 }; return raw; }
  const rvx = raw.cx - st.cx, rvy = raw.cy - st.cy;
  st.vx = st.vx*0.6 + rvx*0.4;
  st.vy = st.vy*0.6 + rvy*0.4;
  const a = SMOOTH, b = 1 - a;
  const s = {
    cx: st.cx*b + raw.cx*a,
    cy: st.cy*b + raw.cy*a,
    ax: [st.ax[0]*b + raw.ax[0]*a, st.ax[1]*b + raw.ax[1]*a],
    ay: [st.ay[0]*b + raw.ay[0]*a, st.ay[1]*b + raw.ay[1]*a],
    r:  st.r*b + raw.r*a,
    pupilR: (st.pupilR||raw.pupilR)*b + raw.pupilR*a,
    irisLuma: (st.irisLuma!=null?st.irisLuma:raw.irisLuma)*0.7 + raw.irisLuma*0.3,
    vx: st.vx, vy: st.vy,
  };
  track[side] = s;
  return { ...s, cx: s.cx + s.vx*PREDICT, cy: s.cy + s.vy*PREDICT };
}

function clamp(v, min, max) { return Math.min(max, Math.max(min, v)); }

// ── 홍채 기하: 4 경계점 → affine 타원 축 벡터 (yaw/pitch/roll 반영) ──────────
function irisGeometry(lm, indices, w, h) {
  const P = i => { const p = lm[indices[i]]; return [w - p.x*w, p.y*h]; }; // 미러 좌표
  const c = P(0);
  const p1 = P(1), p2 = P(2), p3 = P(3), p4 = P(4);
  const v1 = [(p1[0]-p3[0])/2, (p1[1]-p3[1])/2];
  const v2 = [(p2[0]-p4[0])/2, (p2[1]-p4[1])/2];
  let ax, ay;
  if (Math.abs(v1[1]) > Math.abs(v2[1])) { ay = v1; ax = v2; }
  else                                   { ay = v2; ax = v1; }
  if (ay[1] < 0) ay = [-ay[0], -ay[1]];
  if (ax[0] < 0) ax = [-ax[0], -ax[1]];
  const r = (Math.hypot(...ax) + Math.hypot(...ay)) / 2;
  return { cx: c[0], cy: c[1], ax, ay, r, pupilR: r*0.4, irisLuma: 0.34 };
}

// ── 픽셀 분석: 비디오에서 동공(어두운 blob) 검출 → 중심 락온 + 동공반경 + 홍채톤 ─
function refinePupil(geo, w, h) {
  const irisR = geo.r;
  if (irisR < 4 || !video.videoWidth) return geo;
  const R = irisR * 0.9;
  const rawCx = w - geo.cx, rawCy = geo.cy;          // 미러 → 비디오(raw) 좌표
  const sx = Math.max(0, Math.floor(rawCx - R));
  const sy = Math.max(0, Math.floor(rawCy - R));
  const sw = Math.min(video.videoWidth  - sx, Math.ceil(R*2));
  const sh = Math.min(video.videoHeight - sy, Math.ceil(R*2));
  if (sw < 3 || sh < 3) return geo;

  if (sampler.width !== sw || sampler.height !== sh) { sampler.width = sw; sampler.height = sh; }
  let img;
  try {
    sctx.drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh);
    img = sctx.getImageData(0, 0, sw, sh).data;
  } catch { return geo; }

  const lcx = rawCx - sx, lcy = rawCy - sy;
  const searchR = irisR * 0.72;
  // 1차: 최소 휘도 + 홍채 링 평균 휘도(Phase 7)
  let minL = 255, ringSum = 0, ringN = 0;
  for (let py=0; py<sh; py++) for (let px=0; px<sw; px++) {
    const dd = Math.hypot(px-lcx, py-lcy);
    const i=(py*sw+px)*4;
    const L = 0.299*img[i]+0.587*img[i+1]+0.114*img[i+2];
    if (dd <= searchR && L < minL) minL = L;
    if (dd > irisR*0.5 && dd < irisR*0.9) { ringSum += L; ringN++; }
  }
  geo.irisLuma = ringN ? (ringSum/ringN)/255 : 0.34;
  // 2차: 동공 무게중심
  const thr = minL + 28;
  let mx=0, my=0, cnt=0;
  for (let py=0; py<sh; py++) for (let px=0; px<sw; px++) {
    const dd = Math.hypot(px-lcx, py-lcy);
    if (dd > searchR) continue;
    const i=(py*sw+px)*4;
    const L = 0.299*img[i]+0.587*img[i+1]+0.114*img[i+2];
    if (L <= thr) { mx+=px; my+=py; cnt++; }
  }
  if (cnt > 8) {
    const pr = clamp(Math.sqrt(cnt/Math.PI), irisR*0.20, irisR*0.5);
    const mCx = w - (sx + mx/cnt);   // raw → 미러
    const mCy = sy + my/cnt;
    geo.cx = geo.cx*0.5 + mCx*0.5;
    geo.cy = geo.cy*0.5 + mCy*0.5;
    geo.pupilR = pr;
  }
  return geo;
}

// ── 깜빡임/가림 검출 (AR 이식): 눈 감으면 렌즈 숨김 ─────────────────────────
function eyeVisible(lm, contour) {
  const top=lm[contour[4]], bot=lm[contour[12]], left=lm[contour[0]], right=lm[contour[8]];
  const v = Math.hypot(top.x-bot.x, top.y-bot.y);
  const hh = Math.hypot(left.x-right.x, left.y-right.y);
  return (v/hh) >= 0.12; // EAR
}

// ── 텍스처 렌즈 렌더 (affine 타원 + HVID 크기 + 동공 매칭 + 발색 적응) ────────
function drawTextureLens(lensImg, geo, diaMm, contourIndices, landmarks, w, h) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;
  const { cx, cy, ax, ay, r, pupilR, irisLuma } = geo;
  const scale = (diaMm || 14.2) / HVID_MM;

  // Phase 7: 어두운 눈 = 발색 강하게(불투명↑), 밝은 눈 = 은은하게
  const lensAlpha = clamp(0.86 + (0.4 - irisLuma)*0.32, 0.72, 0.96);

  ctx.save();
  // 눈꺼풀 클립
  ctx.beginPath();
  contourIndices.forEach((idx, i) => {
    const p = landmarks[idx];
    const x = w - p.x * w, y = p.y * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.clip();

  ctx.translate(cx, cy);
  ctx.transform(ax[0]*scale, ax[1]*scale, ay[0]*scale, ay[1]*scale, 0, 0);

  // 1. 컨택트 섀도우
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 0.22;
  const cs = ctx.createRadialGradient(0,0,0.92, 0,0,1.12);
  cs.addColorStop(0,"rgba(0,0,0,0)"); cs.addColorStop(0.6,"rgba(30,20,12,0.5)"); cs.addColorStop(1,"rgba(0,0,0,0)");
  ctx.beginPath(); ctx.arc(0,0,1.12,0,Math.PI*2); ctx.fillStyle=cs; ctx.fill();

  // 2. 렌즈 본체 2패스 (source-over + multiply → 홍채 질감 비침)
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = lensAlpha;
  ctx.drawImage(lensImg, -1, -1, 2, 2);
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.24;
  ctx.drawImage(lensImg, -1, -1, 2, 2);
  ctx.globalCompositeOperation = "source-over";

  // 3. 동공 매칭: 검출된 실제 동공을 어둡게 (AR 방식 destination-out)
  const pupilU = clamp((pupilR || r*0.4) / (r*scale), 0.28, 0.55);
  const pg = ctx.createRadialGradient(0,0, pupilU*0.25, 0,0, pupilU);
  pg.addColorStop(0,"rgba(0,0,0,0.82)"); pg.addColorStop(0.62,"rgba(0,0,0,0.42)"); pg.addColorStop(1,"rgba(0,0,0,0)");
  ctx.globalCompositeOperation = "destination-out";
  ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.arc(0,0,pupilU,0,Math.PI*2); ctx.fillStyle=pg; ctx.fill();
  ctx.globalCompositeOperation = "source-over";

  // 4. 이중 캐치라이트
  ctx.globalAlpha = 0.6;
  const spec = (sx,sy,sr,al) => {
    const g = ctx.createRadialGradient(sx,sy,0,sx,sy,sr);
    g.addColorStop(0,`rgba(255,255,255,${al})`); g.addColorStop(1,"rgba(255,255,255,0)");
    ctx.beginPath(); ctx.arc(sx,sy,sr,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
  };
  spec(-0.28,-0.32,0.18,0.9);
  spec( 0.18, 0.12,0.08,0.45);

  ctx.restore();
}

// ── 뷰티용 얼굴 마스크 (AR 이식) ────────────────────────────────────────────
function getTextureTransform() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const cw = beautyCanvas.width || vw, ch = beautyCanvas.height || vh;
  const vAR = vw/vh, cAR = cw/ch;
  let scaleX=1, scaleY=1, offsetX=0, offsetY=0;
  if (cAR > vAR) { scaleY = vAR/cAR; offsetY = (1-scaleY)/2; }
  else           { scaleX = cAR/vAR; offsetX = (1-scaleX)/2; }
  return { scaleX, scaleY, offsetX, offsetY };
}
function updateFaceMask(faces) {
  const S = 512;
  if (faceMaskCanvas.width !== S) { faceMaskCanvas.width = S; faceMaskCanvas.height = S; }
  faceMaskCtx.clearRect(0,0,S,S);
  if (!faces.length) return;
  const t = getTextureTransform();
  faceMaskCtx.save();
  faceMaskCtx.fillStyle = "rgba(255,255,255,1)";
  faces.forEach(lm => {
    const center = { x:(lm[1].x*t.scaleX+t.offsetX)*S, y:(lm[1].y*t.scaleY+t.offsetY)*S };
    const faceHeight = Math.max(0.08, Math.abs((lm[152].y-lm[10].y)*t.scaleY))*S;
    const path = (ex,ey) => {
      faceMaskCtx.beginPath();
      FACE_OVAL.forEach((idx,pi) => {
        const rx=(lm[idx].x*t.scaleX+t.offsetX)*S, ry=(lm[idx].y*t.scaleY+t.offsetY)*S;
        const bias = Math.max(0, ry-center.y)/Math.max(1,faceHeight);
        const x = center.x+(rx-center.x)*(ex+bias*0.025);
        const y = center.y+(ry-center.y)*(ey+bias*0.025);
        pi===0 ? faceMaskCtx.moveTo(x,y) : faceMaskCtx.lineTo(x,y);
      });
      faceMaskCtx.closePath();
    };
    faceMaskCtx.save(); faceMaskCtx.filter="blur(10px)"; faceMaskCtx.globalAlpha=0.34; path(1.01,1.01); faceMaskCtx.fill(); faceMaskCtx.restore();
    faceMaskCtx.save(); faceMaskCtx.filter="blur(3px)";  faceMaskCtx.globalAlpha=0.82; path(0.96,0.96); faceMaskCtx.fill(); faceMaskCtx.restore();
  });
  faceMaskCtx.restore();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
async function loadProducts() {
  const res = await fetch("products.json");
  products = await res.json();
  products.forEach(p => {
    p.diaMm = parseFloat(p.dia) || 14.2;
    if (p.texture) { const img = new Image(); img.src = p.texture; lensImages[p.id] = img; }
  });
  productListEl.innerHTML = "";
  products.forEach(p => {
    const card = document.createElement("div");
    card.className = "product-card";
    card.dataset.id = p.id;
    card.innerHTML = `
      <div class="product-swatch" style="background:${p.swatch}"></div>
      <div class="product-info">
        <p class="product-name">${p.name}</p>
        <p class="product-meta">${p.category} · DIA ${p.dia}</p>
        <p class="product-price">${p.price}</p>
      </div>`;
    card.addEventListener("click", () => selectProduct(p.id));
    productListEl.appendChild(card);
  });
}
function selectProduct(id) {
  activeProduct = products.find(p => p.id === id) || null;
  document.querySelectorAll(".product-card").forEach(el =>
    el.classList.toggle("active", el.dataset.id === id));
}
function setupBeautyButton() {
  const btn = document.createElement("button");
  btn.id = "beauty-btn";
  const labels = ["✨ 뷰티 OFF","✨ 뷰티 약","✨ 뷰티 중","✨ 뷰티 강"];
  const sync = () => { btn.textContent = labels[beautyLevel]; btn.classList.toggle("beauty-on", beautyLevel>0); };
  btn.addEventListener("click", () => { beautyLevel = (beautyLevel+1)%4; sync(); });
  topbar.insertBefore(btn, captureBtn);
  sync();
}

async function setupFaceLandmarker() {
  const fr = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  faceLandmarker = await FaceLandmarker.createFromOptions(fr, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
    numFaces: 2,
    refineLandmarks: true,
  });
}
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  video.play();
  const w = video.videoWidth, h = video.videoHeight;
  canvas.width = w; canvas.height = h;
  beautyCanvas.width = w; beautyCanvas.height = h;
}

// ── Render loop ───────────────────────────────────────────────────────────────
let lastResult = null;
let lastDetect = 0;
const DETECT_INTERVAL = 33;
let running = false;

function processFrame() {
  if (!running) return;
  scheduleNext();
  if (!faceLandmarker || video.readyState < 2) return;

  const w = canvas.width, h = canvas.height;

  const now = performance.now();
  if (now - lastDetect >= DETECT_INTERVAL) {
    lastResult = faceLandmarker.detectForVideo(video, now);
    lastDetect = now;
  }
  const faces = lastResult?.faceLandmarks || [];

  // 1. 뷰티 비디오 → WebGL 캔버스 (밑)
  updateFaceMask(faces);
  softFocusEngine.render(video, BEAUTY_STRENGTH[beautyLevel], faces, faceMaskCanvas);

  // 2. 렌즈 → 오버레이 캔버스 (위, 투명)
  ctx.clearRect(0, 0, w, h);
  if (!faces.length) {
    statusEl.style.display = "block";
    statusEl.textContent = "얼굴을 카메라 앞에 위치시켜 주세요";
    track.left = track.right = null;
    return;
  }
  statusEl.style.display = "none";
  if (!activeProduct || !activeProduct.texture) return;
  const img = lensImages[activeProduct.id];
  if (!img) return;
  const lm = faces[0];
  const dia = activeProduct.diaMm;

  if (eyeVisible(lm, LEFT_EYE_CONTOUR)) {
    const g = trackGeo("left", refinePupil(irisGeometry(lm, LEFT_IRIS, w, h), w, h));
    drawTextureLens(img, g, dia, LEFT_EYE_CONTOUR, lm, w, h);
  }
  if (eyeVisible(lm, RIGHT_EYE_CONTOUR)) {
    const g = trackGeo("right", refinePupil(irisGeometry(lm, RIGHT_IRIS, w, h), w, h));
    drawTextureLens(img, g, dia, RIGHT_EYE_CONTOUR, lm, w, h);
  }
}

const hasRVFC = typeof video.requestVideoFrameCallback === "function";
function scheduleNext() {
  if (hasRVFC) video.requestVideoFrameCallback(processFrame);
  else requestAnimationFrame(processFrame);
}
function startLoop() { if (running) return; running = true; scheduleNext(); }
function stopLoop() { running = false; }
document.addEventListener("visibilitychange", () => { if (document.hidden) stopLoop(); else startLoop(); });

function capture() {
  // 뷰티(WebGL) + 렌즈(오버레이) 합성 캡처
  const out = document.createElement("canvas");
  out.width = canvas.width; out.height = canvas.height;
  const octx = out.getContext("2d");
  octx.drawImage(beautyCanvas, 0, 0);
  octx.drawImage(canvas, 0, 0);
  out.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `lens-tryon-${Date.now()}.png`; a.click();
    URL.revokeObjectURL(url);
  }, "image/png");
}

async function init() {
  captureBtn.addEventListener("click", capture);
  setupBeautyButton();
  await loadProducts();
  if (products.length > 0) selectProduct(products[0].id);
  statusEl.textContent = "모델 로딩 중...";
  await setupFaceLandmarker();
  statusEl.textContent = "카메라 권한을 허용해 주세요";
  try { await setupCamera(); }
  catch (err) { statusEl.textContent = "카메라 접근 실패: " + err.message; return; }
  startLoop();
}

init();
