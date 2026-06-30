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

const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const LEFT_EYE_CONTOUR  = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {};
let lastVideoTime = -1;

// ── Color math ────────────────────────────────────────────────────────────────
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

// ── Mode A: 실제 제품 텍스처 PNG 합성 ────────────────────────────────────────
function drawTextureLens(lensImg, cx, cy, irisR, contourIndices, landmarks, w, h) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;

  // 렌즈 PNG의 링 존 = 이미지의 약 90% → scale=1/0.90=1.11
  // 실제 렌즈는 홍채보다 살짝 크게(+15%) 맞춤
  const scale = 1.28;
  const size  = irisR * scale * 2;

  ctx.save();

  // 눈꺼풀 클리핑
  ctx.beginPath();
  contourIndices.forEach((idx, i) => {
    const p = landmarks[idx];
    const x = w - p.x * w, y = p.y * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.clip();

  // 렌즈 텍스처 합성 (multiply: 렌즈 패턴이 실제 홍채에 자연스럽게 녹아듦)
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 0.90;
  ctx.drawImage(lensImg, cx - size/2, cy - size/2, size, size);

  // 동공 위에 살짝 하이라이트 (광택감)
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = 0.15;
  const hx = cx - irisR*0.18, hy = cy - irisR*0.22;
  const glow = ctx.createRadialGradient(hx, hy, 0, hx, hy, irisR*0.35);
  glow.addColorStop(0, "rgba(255,255,255,1)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.beginPath();
  ctx.arc(cx, cy, irisR, 0, Math.PI*2);
  ctx.fillStyle = glow;
  ctx.fill();

  ctx.restore();
}

// ── Mode B: HSL 픽셀 색상 변환 (색상 데이터만 있을 때) ───────────────────────
function tintIris(cx, cy, irisR, targetRgb, contourIndices, landmarks, w, h) {
  const pupilR = irisR * 0.42;
  const polygon = contourIndices.map(idx => {
    const p = landmarks[idx];
    return [w - p.x * w, p.y * h];
  });
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
      if (!pointInPolygon(wx, wy, polygon)) continue;

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

  // 림벌링
  ctx.save();
  ctx.beginPath();
  contourIndices.forEach((idx,i)=>{ const p=landmarks[idx]; i===0?ctx.moveTo(w-p.x*w,p.y*h):ctx.lineTo(w-p.x*w,p.y*h); });
  ctx.closePath(); ctx.clip();
  ctx.beginPath(); ctx.arc(cx,cy,irisR*0.96,0,Math.PI*2);
  ctx.strokeStyle="rgba(6,4,2,0.82)"; ctx.lineWidth=irisR*0.11; ctx.stroke();
  ctx.globalCompositeOperation="screen";
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

  // 텍스처 이미지 사전 로딩
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

function irisCenterAndRadius(lm, indices, w, h) {
  const c = lm[indices[0]];
  const cx = c.x*w, cy = c.y*h;
  let r = 0;
  for (let i=1; i<indices.length; i++) {
    const p = lm[indices[i]];
    r += Math.sqrt((p.x*w-cx)**2+(p.y*h-cy)**2);
  }
  return { cx, cy, radius: r/(indices.length-1) };
}

// ── Render loop ───────────────────────────────────────────────────────────────
function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (!faceLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = canvas.width, h = canvas.height;
  ctx.save(); ctx.translate(w,0); ctx.scale(-1,1);
  ctx.drawImage(video,0,0,w,h); ctx.restore();

  const result = faceLandmarker.detectForVideo(video, performance.now());
  if (!result.faceLandmarks?.length) {
    statusEl.style.display = "block";
    statusEl.textContent = "얼굴을 카메라 앞에 위치시켜 주세요";
    return;
  }
  statusEl.style.display = "none";
  if (!activeProduct) return;

  const lm = result.faceLandmarks[0];
  const left  = irisCenterAndRadius(lm, LEFT_IRIS,  w, h);
  const right = irisCenterAndRadius(lm, RIGHT_IRIS, w, h);
  left.cx  = w - left.cx;
  right.cx = w - right.cx;

  if (activeProduct.texture && lensImages[activeProduct.id]) {
    // 실제 제품 텍스처 모드
    const img = lensImages[activeProduct.id];
    drawTextureLens(img, left.cx,  left.cy,  left.radius,  LEFT_EYE_CONTOUR,  lm, w, h);
    drawTextureLens(img, right.cx, right.cy, right.radius, RIGHT_EYE_CONTOUR, lm, w, h);
  } else if (activeProduct.color) {
    // HSL 픽셀 색상 모드
    tintIris(left.cx,  left.cy,  left.radius,  activeProduct.color, LEFT_EYE_CONTOUR,  lm, w, h);
    tintIris(right.cx, right.cy, right.radius, activeProduct.color, RIGHT_EYE_CONTOUR, lm, w, h);
  }
}

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
  renderLoop();
}

init();
