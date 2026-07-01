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
const topbar        = document.getElementById("topbar");

// ── 홍채 랜드마크: center + right + bottom + left + top ───────────────────────
const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];

// ── 눈꺼풀 컨투어: MediaPipe 검증된 안구 윤곽점 ──────────────────────────────
const LEFT_EYE_CONTOUR  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const RIGHT_EYE_CONTOUR = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382];

// ── 생물학적 상수: 홍채 가로 지름(HVID) 평균 11.7mm → mm↔px 환산 기준 ───────
const HVID_MM = 11.7;

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {};

// ── 뷰티 필터 레벨 (0=off, 1=약, 2=중, 3=강). 기본 켜짐 ─────────────────────
let beautyLevel = 2;

// ── Phase 5: 추적 상태 (스무딩 + 속도 예측으로 지연 상쇄) ────────────────────
const track = { left: null, right: null };
const SMOOTH  = 0.55; // 위치 반응성 (높을수록 즉각적)
const PREDICT = 0.5;  // 속도 예측 강도 (지연 상쇄, 과하면 오버슛)

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
    vx: st.vx, vy: st.vy,
  };
  track[side] = s;
  // 예측: 현재 속도만큼 앞서 그려 검출/스무딩 지연을 상쇄 → 빠른 움직임 락온
  return { ...s, cx: s.cx + s.vx*PREDICT, cy: s.cy + s.vy*PREDICT };
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

// ── 홍채 기하: 4 경계점 → 축 벡터 2개 (affine 타원, yaw/pitch/roll 반영) ──────
function irisGeometry(lm, indices, w, h) {
  const P = i => { const p = lm[indices[i]]; return [w - p.x*w, p.y*h]; };
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
  return { cx: c[0], cy: c[1], ax, ay, r, pupilR: r*0.4 };
}

// ── Phase 5+6: 픽셀 분석으로 동공 검출 → 중심 락온 + 동공 반경 ───────────────
// 캔버스에 이미 비디오가 그려진 상태에서 눈 ROI의 가장 어두운 blob(동공) 검출
function refinePupil(geo, w, h) {
  const irisR = geo.r;
  if (irisR < 4) return geo;
  const searchR = irisR * 0.72;
  const x0 = Math.max(0, Math.floor(geo.cx - searchR));
  const y0 = Math.max(0, Math.floor(geo.cy - searchR));
  const x1 = Math.min(w, Math.ceil(geo.cx + searchR));
  const y1 = Math.min(h, Math.ceil(geo.cy + searchR));
  const bw = x1-x0, bh = y1-y0;
  if (bw < 3 || bh < 3) return geo;

  let img;
  try { img = ctx.getImageData(x0, y0, bw, bh).data; }
  catch { return geo; }

  // 1차: 검색원 내 최소 휘도 찾기
  let minL = 255;
  for (let py=0; py<bh; py++) {
    for (let px=0; px<bw; px++) {
      const wx=x0+px, wy=y0+py;
      if (Math.hypot(wx-geo.cx, wy-geo.cy) > searchR) continue;
      const i=(py*bw+px)*4;
      const L = 0.299*img[i]+0.587*img[i+1]+0.114*img[i+2];
      if (L < minL) minL = L;
    }
  }
  // 2차: 어두운 픽셀(동공) 무게중심 + 면적
  const thr = minL + 28;
  let sx=0, sy=0, cnt=0;
  for (let py=0; py<bh; py++) {
    for (let px=0; px<bw; px++) {
      const wx=x0+px, wy=y0+py;
      if (Math.hypot(wx-geo.cx, wy-geo.cy) > searchR) continue;
      const i=(py*bw+px)*4;
      const L = 0.299*img[i]+0.587*img[i+1]+0.114*img[i+2];
      if (L <= thr) { sx+=wx; sy+=wy; cnt++; }
    }
  }
  if (cnt > 8) {
    const pcx = sx/cnt, pcy = sy/cnt;
    let pr = Math.sqrt(cnt/Math.PI);
    pr = Math.min(Math.max(pr, irisR*0.20), irisR*0.5);
    // 중심을 검출된 동공 쪽으로 락온 (랜드마크 지연 보정)
    geo.cx = geo.cx*0.5 + pcx*0.5;
    geo.cy = geo.cy*0.5 + pcy*0.5;
    geo.pupilR = pr;
  }
  return geo;
}

// ── Mode A: 실제 텍스처 렌즈 (affine 타원 + DIA 실측 크기 + 동공 매칭) ────────
function drawTextureLens(lensImg, geo, diaMm, contourIndices, landmarks, w, h) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;
  const { cx, cy, ax, ay, r, pupilR } = geo;
  // Phase 6: HVID 11.7mm 기준 실제 DIA 비율로 크기 결정 (오차 최소화)
  const scale = (diaMm || 14.2) / HVID_MM;

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
  // 이후 단위원 좌표계 (텍스처 [-1,1] = 렌즈 반경)

  // 1. 컨택트 섀도우
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 0.22;
  const cs = ctx.createRadialGradient(0, 0, 0.92, 0, 0, 1.12);
  cs.addColorStop(0, "rgba(0,0,0,0)");
  cs.addColorStop(0.6, "rgba(30,20,12,0.5)");
  cs.addColorStop(1, "rgba(0,0,0,0)");
  ctx.beginPath(); ctx.arc(0, 0, 1.12, 0, Math.PI*2);
  ctx.fillStyle = cs; ctx.fill();

  // 2. 렌즈 본체 2패스 (source-over + multiply → 홍채 질감 비침, 깊이감)
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 0.80;
  ctx.drawImage(lensImg, -1, -1, 2, 2);
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.24;
  ctx.drawImage(lensImg, -1, -1, 2, 2);
  ctx.globalCompositeOperation = "source-over";

  // 3. Phase 6: 동공 매칭 — 검출된 실제 동공이 텍스처 구멍보다 크면 부드럽게 어둡게
  const holeU = 0.41;                      // 텍스처 자체 동공 구멍 반경(단위)
  const pupilU = Math.min(0.55, (pupilR || r*0.4) / (r*scale));
  if (pupilU > holeU + 0.02) {
    ctx.globalAlpha = 0.5;
    const pg = ctx.createRadialGradient(0,0, holeU*0.9, 0,0, pupilU);
    pg.addColorStop(0, "rgba(12,9,9,0.75)");
    pg.addColorStop(1, "rgba(12,9,9,0)");
    ctx.beginPath(); ctx.arc(0,0, pupilU, 0, Math.PI*2);
    ctx.fillStyle = pg; ctx.fill();
  }

  // 4. 이중 캐치라이트 (촉촉한 눈)
  ctx.globalAlpha = 0.6;
  const drawSpec = (sx, sy, sr, alpha) => {
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
    g.addColorStop(0, `rgba(255,255,255,${alpha})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI*2);
    ctx.fillStyle = g; ctx.fill();
  };
  drawSpec(-0.28, -0.32, 0.18, 0.9);
  drawSpec( 0.18,  0.12, 0.08, 0.45);

  ctx.restore();
}

// ── Mode B: HSL 픽셀 색상 변환 (Phase 7: 눈동자 색 기반 발색 리얼리티) ────────
function tintIris(geo, targetRgb, contourIndices, landmarks, w, h) {
  const { cx, cy, r: irisR } = geo;
  const pupilR = Math.max(geo.pupilR || irisR*0.4, irisR*0.32);
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

  // Phase 7 1차: 홍채 링 평균 휘도 샘플링 → 밑색 어둡기 파악
  let sumL = 0, nL = 0;
  for (let py=0; py<bh; py+=2) {
    for (let px=0; px<bw; px+=2) {
      const wx=x0+px, wy=y0+py;
      const dist = Math.hypot(wx-cx, wy-cy);
      if (dist < pupilR || dist > irisR*0.98) continue;
      const i=(py*bw+px)*4;
      sumL += (0.299*d[i]+0.587*d[i+1]+0.114*d[i+2])/255;
      nL++;
    }
  }
  const meanL = nL ? sumL/nL : 0.3;
  // 어두운 눈일수록 강하게 발색(불투명↑+밝힘), 밝은 눈은 은은하게(밑색 비침)
  const globalStrength = Math.min(0.98, Math.max(0.62, 0.72 + (0.42 - meanL)*0.9));
  const lift = Math.max(0, (0.42 - meanL)) * 0.7; // 어두운 눈 밝기 보정

  // 2차: 픽셀별 발색
  for (let py=0; py<bh; py++) {
    for (let px=0; px<bw; px++) {
      const wx=x0+px, wy=y0+py;
      const dist = Math.hypot(wx-cx, wy-cy);
      if (dist < pupilR*0.96 || dist > irisR*1.01) continue;
      if (polygon.length > 2 && !pointInPolygon(wx, wy, polygon)) continue;

      const i=(py*bw+px)*4;
      const [pH,pS,pL] = rgbToHsl(d[i],d[i+1],d[i+2]);
      const outerFade = Math.min(1,(irisR-dist)/(irisR*0.13));
      const innerFade = Math.min(1,(dist-pupilR*0.96)/(pupilR*0.30));
      const strength  = outerFade*innerFade*globalStrength;
      const newS = pS+(Math.max(tS,0.45)-pS)*strength;
      const targetL = tL*0.72 + 0.16 + lift;
      const newL = pL+(targetL-pL)*strength;
      const [nr,ng,nb] = hslToRgb(pH+(tH-pH)*strength, newS, Math.min(0.74,newL));
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
  ctx.strokeStyle="rgba(6,4,2,0.80)"; ctx.lineWidth=irisR*0.11; ctx.stroke();
  ctx.globalCompositeOperation="screen";
  ctx.globalAlpha=0.15;
  const hx2=cx-irisR*0.22, hy2=cy-irisR*0.25;
  const g2=ctx.createRadialGradient(hx2,hy2,0,hx2,hy2,irisR*0.38);
  g2.addColorStop(0,"rgba(255,255,255,0.18)"); g2.addColorStop(1,"rgba(255,255,255,0)");
  ctx.beginPath(); ctx.arc(cx,cy,irisR,0,Math.PI*2); ctx.fillStyle=g2; ctx.fill();
  ctx.restore();
}

// ── Phase 8: 뷰티 필터가 적용된 비디오 그리기 ───────────────────────────────
function drawVideoMirrored(w, h) {
  ctx.save(); ctx.translate(w,0); ctx.scale(-1,1);
  ctx.drawImage(video,0,0,w,h); ctx.restore();
}
function drawScene(w, h) {
  if (beautyLevel === 0) { drawVideoMirrored(w, h); return; }
  const L = beautyLevel; // 1~3
  // 톤 보정 (밝기·채도·대비)
  ctx.filter = `brightness(${1+0.035*L}) saturate(${1+0.06*L}) contrast(${1+0.012*L})`;
  drawVideoMirrored(w, h);
  ctx.filter = "none";
  // 스킨 스무딩: 블러 복사본을 soft-light로 → 피부 결 부드럽게(눈은 뒤에서 렌즈로 덮임)
  ctx.save();
  ctx.globalCompositeOperation = "soft-light";
  ctx.globalAlpha = 0.22 + 0.13*L;
  ctx.filter = `blur(${3 + 2*L}px)`;
  drawVideoMirrored(w, h);
  ctx.restore();
  ctx.filter = "none";
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
}

// ── Setup ─────────────────────────────────────────────────────────────────────
async function loadProducts() {
  const res = await fetch("products.json");
  products = await res.json();
  products.forEach(p => {
    p.diaMm = parseFloat(p.dia) || 14.2; // "14.2mm" → 14.2
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

// 뷰티 토글 버튼 (off → 약 → 중 → 강 순환)
function setupBeautyButton() {
  const btn = document.createElement("button");
  btn.id = "beauty-btn";
  const labels = ["✨ 뷰티 OFF", "✨ 뷰티 약", "✨ 뷰티 중", "✨ 뷰티 강"];
  const sync = () => {
    btn.textContent = labels[beautyLevel];
    btn.classList.toggle("beauty-on", beautyLevel > 0);
  };
  btn.addEventListener("click", () => { beautyLevel = (beautyLevel + 1) % 4; sync(); });
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
    numFaces: 1,
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
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ── Render loop ───────────────────────────────────────────────────────────────
let lastResult = null;
let lastDetect = 0;
const DETECT_INTERVAL = 33; // ms ≈ 30Hz
let running = false;

function processFrame() {
  if (!running) return;
  scheduleNext();
  if (!faceLandmarker || video.readyState < 2) return;

  const w = canvas.width, h = canvas.height;

  // 1. 뷰티 필터 적용된 비디오 (렌즈는 이후 위에 샤프하게 그림)
  drawScene(w, h);

  // 2. 얼굴 검출 (30Hz 캡)
  const now = performance.now();
  if (now - lastDetect >= DETECT_INTERVAL) {
    lastResult = faceLandmarker.detectForVideo(video, now);
    lastDetect = now;
  }

  if (!lastResult?.faceLandmarks?.length) {
    statusEl.style.display = "block";
    statusEl.textContent = "얼굴을 카메라 앞에 위치시켜 주세요";
    track.left = track.right = null;
    return;
  }
  statusEl.style.display = "none";
  if (!activeProduct) return;

  const lm = lastResult.faceLandmarks[0];
  // 3. 홍채 기하 → 픽셀 동공 락온 → 추적(스무딩+예측)
  let lRaw = refinePupil(irisGeometry(lm, LEFT_IRIS,  w, h), w, h);
  let rRaw = refinePupil(irisGeometry(lm, RIGHT_IRIS, w, h), w, h);
  const leftGeo  = trackGeo("left",  lRaw);
  const rightGeo = trackGeo("right", rRaw);

  // 4. 렌즈 렌더
  if (activeProduct.texture && lensImages[activeProduct.id]) {
    const img = lensImages[activeProduct.id];
    const dia = activeProduct.diaMm;
    drawTextureLens(img, leftGeo,  dia, LEFT_EYE_CONTOUR,  lm, w, h);
    drawTextureLens(img, rightGeo, dia, RIGHT_EYE_CONTOUR, lm, w, h);
  } else if (activeProduct.color) {
    tintIris(leftGeo,  activeProduct.color, LEFT_EYE_CONTOUR,  lm, w, h);
    tintIris(rightGeo, activeProduct.color, RIGHT_EYE_CONTOUR, lm, w, h);
  }
}

const hasRVFC = typeof video.requestVideoFrameCallback === "function";
function scheduleNext() {
  if (hasRVFC) video.requestVideoFrameCallback(processFrame);
  else requestAnimationFrame(processFrame);
}
function startLoop() { if (running) return; running = true; scheduleNext(); }
function stopLoop() { running = false; }

document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopLoop(); else startLoop();
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
