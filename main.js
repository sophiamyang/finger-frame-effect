import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Demo mode (?demo): synthetic video + fake landmarks, for testing without a camera.
const DEMO = new URLSearchParams(location.search).has("demo");

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const hintEl = document.getElementById("hint");
const toolbar = document.getElementById("toolbar");

// Offscreen canvas for the pixelate effect.
const small = document.createElement("canvas");
const sctx = small.getContext("2d");

const EFFECTS = [
  { id: "pixelate", label: "Pixelate" },
  { id: "blur", label: "Blur" },
  { id: "invert", label: "Invert" },
  { id: "noir", label: "Noir" },
  { id: "glitch", label: "Glitch" },
  { id: "toon", label: "Toon" },
];

// Offscreen canvas for the toon effect (processed at reduced resolution).
const toon = document.createElement("canvas");
const tctx = toon.getContext("2d", { willReadFrequently: true });
const POSTER_LEVELS = 6;
const posterLUT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  posterLUT[i] = Math.round(
    (Math.round((i / 255) * (POSTER_LEVELS - 1)) / (POSTER_LEVELS - 1)) * 255
  );
}
let effect = "toon";

// Smoothed quad corners + presence fade (0..1).
let corners = null;
let presence = 0;
// True while a frame is being shown — relaxes the gesture gate (hysteresis).
let frameActive = false;
// Frames since the quad was last seen; short dropouts hold the last quad.
let lostFrames = 0;
const MAX_LOST_FRAMES = 18;

let landmarker = null;
let lastVideoTime = -1;
let lastResults = null;

function buildToolbar() {
  EFFECTS.forEach((e, i) => {
    const btn = document.createElement("button");
    btn.innerHTML = `<span class="key">${i + 1}</span>${e.label}`;
    btn.dataset.id = e.id;
    if (e.id === effect) btn.classList.add("active");
    btn.addEventListener("click", () => setEffect(e.id));
    toolbar.appendChild(btn);
  });
  window.addEventListener("keydown", (ev) => {
    const idx = parseInt(ev.key, 10) - 1;
    if (idx >= 0 && idx < EFFECTS.length) setEffect(EFFECTS[idx].id);
  });
}

function setEffect(id) {
  effect = id;
  toolbar.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b.dataset.id === id);
  });
}

async function init() {
  buildToolbar();

  let stream;
  if (DEMO) {
    stream = makeDemoStream();
  } else {
    statusText.textContent = "Loading hand tracker…";
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });

    statusText.textContent = "Requesting camera…";
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  }
  video.srcObject = stream;
  await new Promise((res) => (video.onloadedmetadata = res));
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  statusEl.classList.add("hidden");
  requestAnimationFrame(loop);
}

// Draw the (mirrored) camera feed onto any 2d context, filling w x h.
function drawMirrored(c, w, h, dx = 0) {
  c.save();
  c.translate(w, 0);
  c.scale(-1, 1);
  c.drawImage(video, -dx, 0, w, h);
  c.restore();
}

function toPixel(lm) {
  // Mirror x so coordinates match the mirrored canvas.
  return { x: (1 - lm.x) * canvas.width, y: lm.y * canvas.height };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function lerpPt(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// Given landmark sets for exactly two hands, return the 4 frame corners
// (index tip + thumb tip of each hand) ordered around their centroid,
// or null if the hands aren't making an open "L" shape.
function computeQuad(hands) {
  const pts = [];
  for (const lm of hands) {
    const thumb = toPixel(lm[THUMB_TIP]);
    const index = toPixel(lm[INDEX_TIP]);
    // Hand size from wrist -> middle knuckle: stable regardless of which way
    // the fingers point (unlike finger-based measures, which foreshorten).
    const handScale = dist(toPixel(lm[WRIST]), toPixel(lm[MIDDLE_MCP])) + 1;
    // Require thumb and index spread apart (an open "L"). Hysteresis: easy to
    // keep once active, so rotating/foreshortening fingers doesn't drop it.
    const needed = frameActive ? 0.35 : 0.75;
    if (dist(thumb, index) < handScale * needed) return null;
    pts.push(index, thumb);
  }
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  pts.sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
  // Reject degenerate frames (hands overlapping / quad collapsed).
  const minArea = frameActive ? 0.002 : 0.005;
  if (polygonArea(pts) < canvas.width * canvas.height * minArea) return null;
  return pts;
}

// Keep corner identity stable across frames: greedily assign each previous
// corner its nearest target point, so the quad can't twist when the
// angle-sort ordering flips during hand rotation.
function matchToPrev(target, prev) {
  const remaining = [...target];
  return prev.map((c) => {
    let best = 0;
    let bestD = Infinity;
    remaining.forEach((p, i) => {
      const d = dist(c, p);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return remaining.splice(best, 1)[0];
  });
}

function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a / 2);
}

function quadPath(c, q) {
  c.beginPath();
  c.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) c.lineTo(q[i].x, q[i].y);
  c.closePath();
}

function applyEffect(q) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  quadPath(ctx, q);
  ctx.clip();
  ctx.globalAlpha = presence;

  switch (effect) {
    case "pixelate": {
      const factor = 24;
      const sw = Math.max(2, Math.round(w / factor));
      const sh = Math.max(2, Math.round(h / factor));
      if (small.width !== sw || small.height !== sh) {
        small.width = sw;
        small.height = sh;
      }
      drawMirrored(sctx, sw, sh);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(small, 0, 0, sw, sh, 0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      break;
    }
    case "blur": {
      ctx.filter = "blur(14px) saturate(1.1)";
      drawMirrored(ctx, w, h);
      ctx.filter = "none";
      break;
    }
    case "invert": {
      ctx.filter = "invert(1)";
      drawMirrored(ctx, w, h);
      ctx.filter = "none";
      break;
    }
    case "noir": {
      ctx.filter = "grayscale(1) contrast(1.5) brightness(0.95)";
      drawMirrored(ctx, w, h);
      ctx.filter = "none";
      break;
    }
    case "glitch": {
      const t = performance.now() / 1000;
      // Chromatic-aberration ghosts.
      ctx.filter = "saturate(1.6) contrast(1.1)";
      drawMirrored(ctx, w, h);
      ctx.globalAlpha = presence * 0.35;
      ctx.filter = "hue-rotate(120deg)";
      drawMirrored(ctx, w, h, 8 + Math.sin(t * 9) * 5);
      ctx.filter = "hue-rotate(-120deg)";
      drawMirrored(ctx, w, h, -8 - Math.sin(t * 9) * 5);
      ctx.filter = "none";
      // Horizontal slice displacement.
      ctx.globalAlpha = presence;
      const slices = 7;
      for (let i = 0; i < slices; i++) {
        const seed = Math.sin(i * 127.1 + Math.floor(t * 12) * 311.7);
        const sy = ((seed * 0.5 + 0.5) * h) | 0;
        const sliceH = 6 + ((Math.abs(seed) * 26) | 0);
        const dx = (seed * 34) | 0;
        // Source video is mirrored, so sample from the mirrored x range.
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, (sy / h) * video.videoHeight,
          video.videoWidth, (sliceH / h) * video.videoHeight,
          -w + dx, sy, w, sliceH);
        ctx.restore();
      }
      // Scanlines.
      ctx.fillStyle = "rgba(0,0,0,0.16)";
      for (let y = 0; y < h; y += 6) ctx.fillRect(0, y, w, 2);
      break;
    }
    case "toon": {
      drawToon(w, h);
      break;
    }
  }

  ctx.restore();
}

// ---- Toon: cel-shaded cartoon version of the live feed ----
// Smooth the feed, quantize colors into flat bands, then draw dark outline
// strokes where Sobel edge magnitude is high.
function drawToon(w, h) {
  const tw = 320;
  const th = Math.max(2, Math.round((tw * h) / w));
  if (toon.width !== tw || toon.height !== th) {
    toon.width = tw;
    toon.height = th;
  }
  // Slight blur approximates bilateral smoothing; saturation sells the
  // "animated" look.
  tctx.filter = "saturate(1.6) blur(0.6px) brightness(1.05)";
  drawMirrored(tctx, tw, th);
  tctx.filter = "none";

  const imgData = tctx.getImageData(0, 0, tw, th);
  const d = imgData.data;

  // Luminance (pre-posterize) for edge detection.
  const lum = new Float32Array(tw * th);
  for (let i = 0, p = 0; i < lum.length; i++, p += 4) {
    lum[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  }

  // Posterize colors into flat cel bands.
  for (let p = 0; p < d.length; p += 4) {
    d[p] = posterLUT[d[p]];
    d[p + 1] = posterLUT[d[p + 1]];
    d[p + 2] = posterLUT[d[p + 2]];
  }

  // Sobel edges -> dark outlines.
  for (let y = 1; y < th - 1; y++) {
    for (let x = 1; x < tw - 1; x++) {
      const i = y * tw + x;
      const gx =
        -lum[i - tw - 1] - 2 * lum[i - 1] - lum[i + tw - 1] +
        lum[i - tw + 1] + 2 * lum[i + 1] + lum[i + tw + 1];
      const gy =
        -lum[i - tw - 1] - 2 * lum[i - tw] - lum[i - tw + 1] +
        lum[i + tw - 1] + 2 * lum[i + tw] + lum[i + tw + 1];
      const mag = Math.abs(gx) + Math.abs(gy);
      if (mag > 90) {
        const p = i * 4;
        d[p] *= 0.18;
        d[p + 1] *= 0.18;
        d[p + 2] *= 0.18;
      }
    }
  }

  tctx.putImageData(imgData, 0, 0);
  ctx.drawImage(toon, 0, 0, tw, th, 0, 0, w, h);
}

function drawFrameOutline(q) {
  const t = performance.now() / 1000;
  ctx.save();
  ctx.globalAlpha = presence;

  quadPath(ctx, q);
  ctx.setLineDash([10, 8]);
  // Marching ants: slide the dash pattern along the outline.
  ctx.lineDashOffset = -t * 40;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.shadowBlur = 6;
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;
  ctx.shadowBlur = 0;
  q.forEach((p, i) => {
    const r = 7 + Math.sin(t * 3 + i * 1.5) * 1.5;
    // Soft expanding halo behind each corner dot.
    const halo = (t * 0.8 + i * 0.25) % 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + halo * 14, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${0.5 * (1 - halo) * presence})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

function loop() {
  const w = canvas.width;
  const h = canvas.height;

  // Base layer: mirrored camera feed.
  drawMirrored(ctx, w, h);

  // Run detection once per new video frame.
  if (DEMO) {
    lastResults = { landmarks: fakeHands(performance.now() / 1000) };
  } else if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    lastResults = landmarker.detectForVideo(video, performance.now());
  }

  let targetQuad = null;
  if (lastResults && lastResults.landmarks && lastResults.landmarks.length === 2) {
    targetQuad = computeQuad(lastResults.landmarks);
  }

  if (targetQuad) {
    lostFrames = 0;
    frameActive = true;
    if (!corners) {
      corners = targetQuad;
    } else {
      const matched = matchToPrev(targetQuad, corners);
      corners = corners.map((c, i) => lerpPt(c, matched[i], 0.4));
    }
    presence = Math.min(1, presence + 0.12);
  } else if (corners && ++lostFrames <= MAX_LOST_FRAMES) {
    // Brief tracking dropout: hold the last quad instead of fading.
    presence = Math.min(1, presence + 0.12);
  } else {
    presence = Math.max(0, presence - 0.05);
    if (presence === 0) {
      corners = null;
      frameActive = false;
    }
  }

  if (corners && presence > 0.01) {
    applyEffect(corners);
    drawFrameOutline(corners);
  }

  hintEl.classList.toggle("hidden", presence > 0.5);

  requestAnimationFrame(loop);
}

// ---- Demo mode helpers ----

function makeDemoStream() {
  const demo = document.createElement("canvas");
  demo.width = 1280;
  demo.height = 720;
  const d = demo.getContext("2d");
  function paint() {
    const t = performance.now() / 1000;
    const g = d.createLinearGradient(0, 0, demo.width, demo.height);
    g.addColorStop(0, "#1c2a4a");
    g.addColorStop(1, "#3a1c4a");
    d.fillStyle = g;
    d.fillRect(0, 0, demo.width, demo.height);
    for (let i = 0; i < 6; i++) {
      const x = demo.width * (0.15 + 0.14 * i) + Math.sin(t * 0.8 + i) * 60;
      const y = demo.height * 0.5 + Math.cos(t * 0.6 + i * 1.7) * 160;
      d.beginPath();
      d.arc(x, y, 50 + 18 * Math.sin(t + i), 0, Math.PI * 2);
      d.fillStyle = `hsl(${(i * 60 + t * 30) % 360}, 75%, 62%)`;
      d.fill();
    }
    d.fillStyle = "rgba(255,255,255,0.9)";
    d.font = "bold 56px sans-serif";
    d.textAlign = "center";
    // Draw mirrored so it reads correctly after the canvas flips it back.
    d.save();
    d.translate(demo.width, 0);
    d.scale(-1, 1);
    d.fillText("DEMO FEED", demo.width / 2, demo.height / 2);
    d.restore();
    requestAnimationFrame(paint);
  }
  paint();
  return demo.captureStream(30);
}

function fakeHand(indexTip, thumbTip, indexMcp) {
  const lm = Array.from({ length: 21 }, () => ({ ...indexMcp, z: 0 }));
  lm[INDEX_TIP] = { ...indexTip, z: 0 };
  lm[THUMB_TIP] = { ...thumbTip, z: 0 };
  lm[INDEX_MCP] = { ...indexMcp, z: 0 };
  return lm;
}

function fakeHands(t) {
  const ox = Math.sin(t * 0.9) * 0.02;
  const oy = Math.cos(t * 0.7) * 0.02;
  return [
    fakeHand(
      { x: 0.74 + ox, y: 0.26 + oy },
      { x: 0.8 + ox, y: 0.56 + oy },
      { x: 0.75 + ox, y: 0.4 + oy }
    ),
    fakeHand(
      { x: 0.26 - ox, y: 0.64 - oy },
      { x: 0.2 - ox, y: 0.34 - oy },
      { x: 0.25 - ox, y: 0.5 - oy }
    ),
  ];
}

init().catch((err) => {
  console.error(err);
  statusEl.classList.remove("hidden");
  statusEl.querySelector(".spinner")?.remove();
  statusText.textContent =
    err.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access and reload."
      : `Failed to start: ${err.message}`;
});
