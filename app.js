import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const productListEl = document.getElementById("product-list");
const captureBtn = document.getElementById("capture-btn");

// Iris landmark indices (MediaPipe FaceMesh, refineLandmarks = true)
// center, then 4 boundary points around the iris
const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];

let faceLandmarker = null;
let products = [];
let activeProduct = null;
const lensImages = {}; // id -> HTMLImageElement
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
  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
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

function drawLensOnEye(lensImg, cx, cy, radius) {
  if (!lensImg.complete || lensImg.naturalWidth === 0) return;
  const scale = 2.1; // lens slightly larger than detected iris radius for natural coverage
  const size = radius * scale * 2;
  ctx.drawImage(lensImg, cx - size / 2, cy - size / 2, size, size);
}

function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (!faceLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

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

  const left = irisCenterAndRadius(landmarks, LEFT_IRIS, w, h);
  const right = irisCenterAndRadius(landmarks, RIGHT_IRIS, w, h);

  drawLensOnEye(lensImg, left.cx, left.cy, left.radius);
  drawLensOnEye(lensImg, right.cx, right.cy, right.radius);
}

function capture() {
  const out = document.createElement("canvas");
  out.width = canvas.width;
  out.height = canvas.height;
  const octx = out.getContext("2d");
  // mirror to match what the user sees on screen
  octx.translate(out.width, 0);
  octx.scale(-1, 1);
  octx.drawImage(video, 0, 0, out.width, out.height);
  octx.drawImage(canvas, 0, 0, out.width, out.height);

  out.toBlob((blob) => {
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
