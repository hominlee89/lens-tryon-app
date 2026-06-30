import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video    = document.getElementById("video");
const canvas   = document.getElementById("overlay");
const ctx      = canvas.getContext("2d");
video.style.display = "none";
const statusEl      = document.getElementById("status");
const productListEl = document.getElementById("product-list");
const captureBtn    = document.getElementById("capture-btn");

// ── 홍채 랜드마크: center + right + bottom + left + top ───────────────────────
const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];

// ── 눈꺼풀 컨투어: MediaPipe 검증된 안구 윤곽점 ──────────────────────────────
const LEFT_EYE_CONTOUR  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const RIGHT_EYE_CONTOUR = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382];

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {};
let lastVideoTime = -1;

// ── 시간적 스무딩 상태 (떨림 제거) ──────────────────────────────────────────
const smoothState = { left: null, right: null };
const SMOOTH = 0.45; // 0=고정(지연 큼) ~ 1=즉시(떨림). 0.45=안정+반응 균형

function smoothGeo(side, geo) {
  const prev = smoothState[side];
  if (!prev) { smoothState[side] = geo; return geo; }
  const a = SMOOTH, b = 1 - a;
  const out = {
    cx: prev.cx*b + geo.cx*a,
    cy: prev.cy*b + geo.cy*a,
    ax: [prev.ax[0]*b + geo.ax[0]*a, prev.ax[1]*b + geo.ax[1]*a],
    ay: [prev.ay[0]*b + geo.ay[0]*a, prev.ay[1]*b + geo.ay[1]*a],
    r:  prev.r*b + geo.r*a,
  };
  smoothState[side] = out;
  return out;
}

// ── RGB ↔ HSL ─────────────────────────────────────────────────────────────────
function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r,g,b), min = Math.min(r,g,b), l = (max+min)/2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d/(2-max-min) : d/(max+min);
  let h;
  if (max===r)      h = ((g-b)/d + (g<b?6:0))/6;
  else if (max===g) h = ((b-r)/d + 2)/6;
  else              h = ((r-g)/d + 4)/6;
  return [h, s, l];
}
function hslToRgb(h, s, l) {
  if (s === 0) { const v = Math.round(l*255); return [v,v,v]; }
  const q = l<0.5 ? l*(1+s) : l+s-l*s, p = 2*l-q;
  const hue = t => {
    if(t<0)t+=1; if(t>1)t-=1;
    if(t<1/6) return p+(q-p)*6*t;
    if(t<1/2) return q;
    if(t<2/3) return p+(q-p)*(2/3-t)*6;
    return p;
  };
  return [Math.round(hue(h+1/3)*255), Math.round(hue(h)*255), Math.round(hue(h-1/3)*255)];
}
function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i=0, j=poly.length-1; i<poly.length; j=i++) {
    const [xi,yi]=poly[i],[xj,yj]=poly[j];
    if(((yi>py)!==(yj>py))&&(px<(xj-xi)*(py-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

// ── 홍채 기하 계산: 4 경계점 → 축 벡터 2개 (affine 타원) ─────────────────────
// 고개를 돌리면 홍채 경계점이 원근감으로 찌그러짐 → 축 벡터가 그대로 yaw/pitch/roll 반영
function irisGeometry(lm, indices, w, h) {
  // 화면 좌표(미러 적용)로 변환
  const P = i => { const p = lm[indices[i]]; return [w - p.x*w, p.y*h]; };
  const c = P(0);
  // indices[1..4]: 원 둘레의 4점, 마주보는 쌍 = (1,3), (2,4)
  const p1 = P(1), p2 = P(2), p3 = P(3), p4 = P(4);
  const v1 = [(p1[0]-p3[0])/2, (p1[1]-p3[1])/2];
  const v2 = [(p2[0]-p4[0])/2, (p2[1]-p4[1])/2];
  // 더 세로에 가까운 벡터를 ay(수직축), 나머지를 ax(수평축)로
  let ax, ay;
  if (Math.abs(v1[1]) > Math.abs(v2[1])) { ay = v1; ax = v2; }
  else                                   { ay = v2; ax = v1; }
  // 방향 부호 고정: ay는 아래(+y), ax는 오른쪽(+x) → 텍스처 상하/좌우 뒤집힘 방지
  if (ay[1] < 0) ay = [-ay[0], -ay[1]];
  if (ax[0] < 0) ax = [-ax[0], -ax[1]];
  const lenA = Math.hypot(ax[0], ax[1]);
  const lenB = Math.hypot(ay[0], ay[1]);
  return { cx: c[0], cy: c[1], ax, ay, r: (lenA + lenB) / 2 };
}

// ── Mode A: 실제 텍스처 렌즈 (affine 타원 — 얼굴 각도 자동 보정) ────────────
function drawTextureLens(lensImg, geo, contourIndices, landmarks, w, h) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;
  const { cx, cy, ax, ay, r } = geo;
  const scale = 1.28; // 홍채보다 살짝 크게

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

  // 축 벡터로 affine 변환 구성: 단위원 → 실제 홍채 타원(각도 포함)
  ctx.translate(cx, cy);
  ctx.transform(ax[0]*scale, ax[1]*scale, ay[0]*scale, ay[1]*scale, 0, 0);
  // 이후 모든 그리기는 단위원 좌표계 → 타원/회전 자동 적용

  // 1. 컨택트 섀도우 (렌즈 가장자리 바깥쪽 옅은 그림자 → 눈 위에 "앉은" 깊이감)
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 0.22;
  const cs = ctx.createRadialGradient(0, 0, 0.92, 0, 0, 1.12);
  cs.addColorStop(0, "rgba(0,0,0,0)");
  cs.addColorStop(0.6, "rgba(30,20,12,0.5)");
  cs.addColorStop(1, "rgba(0,0,0,0)");
  ctx.beginPath(); ctx.arc(0, 0, 1.12, 0, Math.PI*2);
  ctx.fillStyle = cs; ctx.fill();

  // 2. 렌즈 텍스처 본체
  ctx.globalAlpha = 0.92;
  ctx.drawImage(lensImg, -1, -1, 2, 2);

  // 3. 이중 캐치라이트 (점광원 반사 — 촉촉한 눈). 단위원 좌표라 같이 변형됨
  ctx.globalAlpha = 0.6;
  const drawSpec = (sx, sy, sr, alpha) => {
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
    g.addColorStop(0, `rgba(255,255,255,${alpha})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI*2);
    ctx.fillStyle = g; ctx.fill();
  };
  drawSpec(-0.28, -0.32, 0.18, 0.9);   // 주 반사 (좌상단)
  drawSpec( 0.18,  0.12, 0.08, 0.45);  // 보조 반사 (우하단, 작게)

  ctx.restore();
}

// ── Mode B: HSL 픽셀 색상 변환 ───────────────────────────────────────────────
function tintIris(cx, cy, irisR, targetRgb, contourIndices, landmarks, w, h) {
  const pupilR = irisR * 0.42;
  const polygon = contourIndices
    .map(idx => { const p = landmarks[idx]; return [w - p.x*w, p.y*h]; })
    .filter(([x,y]) => x > 0 && y > 0 && x < w && y < h);
  const [tH, tS, tL] = rgbToHsl(...targetRgb);

  const x0 = Math.max(0, Math.floor(cx - irisR - 2));
  const y0 = Math.max(0, Math.floor(cy - irisR - 2));
  const x1 = Math.min(w, Math.ceil(cx + irisR + 2));
  const y1 = Math.min(h, Math.ceil(cy + irisR + 2));
  const bw = x1-x0, bh = y1-y0;
  if (bw<=0||bh<=0) return;

  const imageData = ctx.getImageData(x0, y0, bw, bh);
  const d = imageData.data;

  for (let py=0; py<bh; py++) {
    for (let px=0; px<bw; px++) {
      const wx=x0+px, wy=y0+py;
      const dx=wx-cx, dy=wy-cy;
      const dist = Math.sqrt(dx*dx+dy*dy);
      if (dist < pupilR*0.88 || dist > irisR*1.01) continue;
      if (polygon.length > 2 && !pointInPolygon(wx, wy, polygon)) continue;

      const i = (py*bw+px)*4;
      const [pH,pS,pL] = rgbToHsl(d[i],d[i+1],d[i+2]);
      const outerFade = Math.min(1,(irisR-dist)/(irisR*0.13));
      const innerFade = Math.min(1,(dist-pupilR*0.88)/(pupilR*0.30));
      const strength  = outerFade*innerFade*0.92;
      const newS = pS+(Math.max(tS,0.45)-pS)*strength;
      const darkBoost = (1-pL)*0.55;
      const targetL = tL*0.75+0.18;
      const newL = pL+(targetL-pL+darkBoost)*strength;
      const [nr,ng,nb] = hslToRgb(pH+(tH-pH)*strength, newS, Math.min(0.72,newL));
      d[i]=nr; d[i+1]=ng; d[i+2]=nb;
    }
  }
  ctx.putImageData(imageData, x0, y0);

  // 림벌링 + 하이라이트
  ctx.save();
  if (polygon.length > 2) {
    ctx.beginPath();
    polygon.forEach(([x,y],i)=> i===0?ctx.moveTo(x,y):ctx.lineTo(x,y));
    ctx.closePath(); ctx.clip();
  }
  ctx.beginPath(); ctx.arc(cx,cy,irisR*0.96,0,Math.PI*2);
  ctx.strokeStyle="rgba(6,4,2,0.82)"; ctx.lineWidth=irisR*0.11; ctx.stroke();
  ctx.globalCompositeOperation="screen";
  ctx.globalAlpha=0.15;
  const hx2=cx-irisR*0.22, hy2=cy-irisR*0.25;
  const g2=ctx.createRadialGradient(hx2,hy2,0,hx2,hy2,irisR*0.38);
  g2.addColorStop(0,"rgba(255,255,255,0.18)"); g2.addColorStop(1,"rgba(255,255,255,0)");
  ctx.beginPath(); ctx.arc(cx,cy,irisR,0,Math.PI*2); ctx.fillStyle=g2; ctx.fill();
  ctx.restore();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
async function loadProducts() {
  const res = await fetch("products.json");
  products = await res.json();
  products.forEach(p => {
    if (p.texture) {
      const img = new Image();
      img.src = p.texture;
      lensImages[p.id] = img;
    }
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
    numFaces: 1,
    refineLandmarks: true,
  });
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => { video.onloadedmetadata = r; });
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ── Render loop ───────────────────────────────────────────────────────────────
// 검출(ML)과 렌더를 분리: 검출은 ~30Hz로 캡, 렌더는 매 비디오 프레임
let lastResult = null;
let lastDetect = 0;
const DETECT_INTERVAL = 33; // ms ≈ 30Hz. ML 부하/발열 완화
let running = false;

function processFrame() {
  if (!running) return;
  scheduleNext();
  if (!faceLandmarker || video.readyState < 2) return;

  const w = canvas.width, h = canvas.height;

  // 비디오 그리기 (미러)
  ctx.save(); ctx.translate(w,0); ctx.scale(-1,1);
  ctx.drawImage(video,0,0,w,h); ctx.restore();

  // 검출은 간격 캡 — 그 사이 프레임은 직전 결과 재사용 (스무딩이 연속성 보완)
  const now = performance.now();
  if (now - lastDetect >= DETECT_INTERVAL) {
    lastResult = faceLandmarker.detectForVideo(video, now);
    lastDetect = now;
  }

  if (!lastResult?.faceLandmarks?.length) {
    statusEl.style.display = "block";
    statusEl.textContent = "얼굴을 카메라 앞에 위치시켜 주세요";
    smoothState.left = smoothState.right = null; // 얼굴 사라지면 스무딩 리셋
    return;
  }
  statusEl.style.display = "none";
  if (!activeProduct) return;

  const lm = lastResult.faceLandmarks[0];
  const leftGeo  = smoothGeo("left",  irisGeometry(lm, LEFT_IRIS,  w, h));
  const rightGeo = smoothGeo("right", irisGeometry(lm, RIGHT_IRIS, w, h));

  if (activeProduct.texture && lensImages[activeProduct.id]) {
    const img = lensImages[activeProduct.id];
    drawTextureLens(img, leftGeo,  LEFT_EYE_CONTOUR,  lm, w, h);
    drawTextureLens(img, rightGeo, RIGHT_EYE_CONTOUR, lm, w, h);
  } else if (activeProduct.color) {
    tintIris(leftGeo.cx,  leftGeo.cy,  leftGeo.r,  activeProduct.color, LEFT_EYE_CONTOUR,  lm, w, h);
    tintIris(rightGeo.cx, rightGeo.cy, rightGeo.r, activeProduct.color, RIGHT_EYE_CONTOUR, lm, w, h);
  }
}

// 비디오 프레임 콜백 우선 (정확한 동기 + 배터리 절약), 미지원 시 rAF 폴백
const hasRVFC = typeof video.requestVideoFrameCallback === "function";
function scheduleNext() {
  if (hasRVFC) video.requestVideoFrameCallback(processFrame);
  else requestAnimationFrame(processFrame);
}
function startLoop() {
  if (running) return;
  running = true;
  scheduleNext();
}
function stopLoop() { running = false; }

// 탭이 백그라운드로 가면 루프 정지 (배터리/발열 절약), 복귀 시 재개
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopLoop();
  else startLoop();
});

function capture() {
  canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `lens-tryon-${Date.now()}.png`; a.click();
    URL.revokeObjectURL(url);
  }, "image/png");
}

async function init() {
  captureBtn.addEventListener("click", capture);
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
