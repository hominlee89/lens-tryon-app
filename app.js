import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
video.style.display = "none";
const statusEl = document.getElementById("status");
const productListEl = document.getElementById("product-list");
const captureBtn = document.getElementById("capture-btn");

// Iris landmarks (refineLandmarks=true): center + 4 boundary points
const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];

// Eye-opening contour landmarks (traces upper + lower eyelid boundary)
const LEFT_EYE_CONTOUR  = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const RIGHT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {};
let lastVideoTime = -1;

async function loadProducts() {
  const res = await fetch("products.json");
  products = await res.json();
  products.forEach((p) => {
    const img = new Image();
    img.src = p.texture;
    lensImages[p.id] = img;
  });
  productListEl.innerHTML = "";
  products.forEach((p) => {
    const card = document.createElement("div");
    card.className = "product-card";
    card.dataset.id = p.id;
    card.innerHTML = `
      <div class="product-swatch" style="background:${p.swatch}"></div>
      <div class="product-info">
        <p class="product-name">${p.name}</p>
        <p class="product-meta">${p.category} · DIA ${p.dia}</p>
        <p class="product-price">${p.price}</p>
      </div>
    `;
    card.addEventListener("click", () => selectProduct(p.id));
    productListEl.appendChild(card);
  });
}

function selectProduct(id) {
  activeProduct = products.find((p) => p.id === id) || null;
  document.querySelectorAll(".product-card").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });
}

async function setupFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
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
  await new Promise((resolve) => { video.onloadedmetadata = () => resolve(); });
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function irisCenterAndRadius(landmarks, indices, w, h) {
  const center = landmarks[indices[0]];
  const cx = center.x * w;
  const cy = center.y * h;
  let radius = 0;
  for (let i = 1; i < indices.length; i++) {
    const p = landmarks[indices[i]];
    const dx = p.x * w - cx;
    const dy = p.y * h - cy;
    radius += Math.sqrt(dx * dx + dy * dy);
  }
  radius /= indices.length - 1;
  return { cx, cy, radius };
}

// Clip canvas to eye-opening shape, draw lens inside, then restore
function drawLensWithEyeClip(lensImg, irisX, irisY, radius, contourIndices, landmarks, w, h) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;

  // Lens PNG: colored zone = 41% of half-image → scale = 1/0.82 = 1.22 to match iris
  const scale = 1.22;
  const size = radius * scale * 2;

  ctx.save();

  // Build eyelid clipping path (mirrored x)
  ctx.beginPath();
  contourIndices.forEach((idx, i) => {
    const pt = landmarks[idx];
    const x = w - pt.x * w;
    const y = pt.y * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.clip(); // lens only renders inside the eye opening

  ctx.globalAlpha = 0.82; // semi-transparent so real iris/pupil shows through
  ctx.drawImage(lensImg, irisX - size / 2, irisY - size / 2, size, size);

  ctx.restore();
}

function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (!faceLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = canvas.width;
  const h = canvas.height;

  // Draw mirrored video as background
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const result = faceLandmarker.detectForVideo(video, performance.now());

  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    statusEl.style.display = "block";
    statusEl.textContent = "얼굴을 카메라 앞에 위치시켜 주세요";
    return;
  }
  statusEl.style.display = "none";

  if (!activeProduct) return;
  const lensImg = lensImages[activeProduct.id];
  if (!lensImg) return;

  const landmarks = result.faceLandmarks[0];

  const left  = irisCenterAndRadius(landmarks, LEFT_IRIS,  w, h);
  const right = irisCenterAndRadius(landmarks, RIGHT_IRIS, w, h);

  // Mirror iris centers to match flipped video
  left.cx  = w - left.cx;
  right.cx = w - right.cx;

  // Draw each lens clipped to its eyelid opening
  drawLensWithEyeClip(lensImg, left.cx,  left.cy,  left.radius,  LEFT_EYE_CONTOUR,  landmarks, w, h);
  drawLensWithEyeClip(lensImg, right.cx, right.cy, right.radius, RIGHT_EYE_CONTOUR, landmarks, w, h);
}

function capture() {
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lens-tryon-${Date.now()}.png`;
    a.click();
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
  try {
    await setupCamera();
  } catch (err) {
    statusEl.textContent = "카메라 접근 실패: " + err.message;
    return;
  }

  renderLoop();
}

init();
