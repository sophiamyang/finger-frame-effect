// MediaPipe is vendored locally (see vendor/mediapipe/) so the page loads no
// third-party code — important since users may store an API key in the browser.
import {
  HandLandmarker,
  FilesetResolver,
} from "./vendor/mediapipe/vision_bundle.mjs";

const WASM_URL = "./vendor/mediapipe/wasm";
const MODEL_URL = "./vendor/mediapipe/hand_landmarker.task";

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
  { id: "ai", label: "AI ✨" },
];

// ---- AI effect (Gemini image-to-image) state ----
const AI_MODEL = "gemini-2.5-flash-image";
const AI_STYLES = {
  movie3d:
    "Transform the person in this image into a 3D animated movie character " +
    "(stylized CGI animation look, expressive big eyes, soft lighting).",
  anime:
    "Redraw this image as a hand-drawn anime illustration with clean line " +
    "art, cel shading, and vibrant colors.",
  clay:
    "Transform the person in this image into a claymation stop-motion clay " +
    "character with visible clay texture.",
  pixel:
    "Redraw this image as detailed retro pixel art in a 16-bit video game style.",
  watercolor:
    "Repaint this image as a soft watercolor painting with loose brushwork " +
    "and gentle color bleeds.",
};
// Always appended so the result stays registered with the live feed.
const AI_PROMPT_SUFFIX =
  " This is critical: preserve the EXACT position, scale, and framing of " +
  "every element — the face, body, hands, and background must stay in " +
  "precisely the same place and size as the original photo. Do not enlarge, " +
  "recenter, zoom, or crop anything. Same pose, same expression, same " +
  "clothing colors, same background. Return only the transformed image.";
let aiStyle = localStorage.getItem("ai-style") || "movie3d";
let aiCustomPrompt = localStorage.getItem("ai-style-custom") || "";

function aiPrompt() {
  const style =
    aiStyle === "custom" && aiCustomPrompt.trim()
      ? aiCustomPrompt.trim()
      : AI_STYLES[aiStyle] || AI_STYLES.movie3d;
  return style + AI_PROMPT_SUFFIX;
}
let apiKey =
  localStorage.getItem("gemini-key") || sessionStorage.getItem("gemini-key") || "";
let aiState = "idle"; // idle | generating | ready | error
let aiImage = null;
let aiError = "";
let aiLastGen = 0;
// Re-stylize every few seconds while the frame is held, so the window keeps
// up as the person moves. Each refresh is one image request.
const AI_REFRESH_MS = 6000;
let stableFrames = 0;
let lastRawQuad = null;
// Full-res snapshot canvas for captures.
const snap = document.createElement("canvas");
const snapCtx = snap.getContext("2d");

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
  if (id === "ai" && !apiKey) {
    document.getElementById("key-panel").classList.remove("hidden");
  }
}

function setupKeyPanel() {
  const btn = document.getElementById("key-btn");
  const panel = document.getElementById("key-panel");
  const input = document.getElementById("key-input");
  const remember = document.getElementById("key-remember");
  const styleSelect = document.getElementById("style-select");
  const styleCustom = document.getElementById("style-custom");

  input.value = apiKey;
  remember.checked = !!localStorage.getItem("gemini-key");
  styleSelect.value = aiStyle;
  styleCustom.value = aiCustomPrompt;
  styleCustom.classList.toggle("hidden", aiStyle !== "custom");

  btn.addEventListener("click", () => panel.classList.toggle("hidden"));
  styleSelect.addEventListener("change", () => {
    styleCustom.classList.toggle("hidden", styleSelect.value !== "custom");
  });
  document.getElementById("key-save").addEventListener("click", () => {
    apiKey = input.value.trim();
    localStorage.removeItem("gemini-key");
    sessionStorage.removeItem("gemini-key");
    if (apiKey) {
      (remember.checked ? localStorage : sessionStorage).setItem("gemini-key", apiKey);
    }
    aiStyle = styleSelect.value;
    aiCustomPrompt = styleCustom.value;
    localStorage.setItem("ai-style", aiStyle);
    localStorage.setItem("ai-style-custom", aiCustomPrompt);
    // New style means the cached result is stale — regenerate on next framing.
    aiImage = null;
    aiState = "idle";
    aiError = "";
    panel.classList.add("hidden");
  });
  document.getElementById("key-clear").addEventListener("click", () => {
    apiKey = "";
    input.value = "";
    localStorage.removeItem("gemini-key");
    sessionStorage.removeItem("gemini-key");
  });
}

async function init() {
  buildToolbar();
  setupKeyPanel();

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
    case "ai": {
      if (aiImage) {
        // The stylized image is a full-frame render aligned with the camera
        // feed — draw it screen-aligned so the finger frame acts as a window
        // revealing it, staying registered with the real scene as hands move.
        ctx.drawImage(aiImage, 0, 0, w, h);
      } else {
        // Waiting / generating: blurred feed + shimmer sweep + status text.
        ctx.filter = "blur(10px) brightness(0.85) saturate(1.1)";
        drawMirrored(ctx, w, h);
        ctx.filter = "none";
        drawShimmer(q);
        let label;
        if (!apiKey) label = "🔑 Add your Gemini key (top right)";
        else if (aiState === "generating") label = "✨ Dreaming you up…";
        else if (aiState === "error") label = "⚠️ " + aiError;
        else label = "✨ Hold the frame steady…";
        drawQuadLabel(q, label);
      }
      break;
    }
  }

  ctx.restore();
}

function quadBBox(q) {
  const xs = q.map((p) => p.x);
  const ys = q.map((p) => p.y);
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  };
}

function drawShimmer(q) {
  const b = quadBBox(q);
  const w = b.x1 - b.x0;
  const t = (performance.now() / 1400) % 1;
  const gx = b.x0 - w * 0.3 + t * w * 1.6;
  const grad = ctx.createLinearGradient(gx - w * 0.2, 0, gx + w * 0.2, 0);
  grad.addColorStop(0, "rgba(255,255,255,0)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.28)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(b.x0, b.y0, w, b.y1 - b.y0);
}

function drawQuadLabel(q, text) {
  const cx = q.reduce((s, p) => s + p.x, 0) / 4;
  const cy = q.reduce((s, p) => s + p.y, 0) / 4;
  ctx.font = `600 ${Math.round(canvas.width / 55)}px -apple-system, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.shadowColor = "rgba(0,0,0,0.7)";
  ctx.shadowBlur = 8;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fillText(text, cx, cy);
  ctx.shadowBlur = 0;
}

// ---- AI generation (Gemini image-to-image) ----

// Capture the entire (mirrored) camera frame as base64 JPEG. The whole scene
// is stylized so the result stays aligned with the feed and the finger frame
// can act as a moving window over it.
function captureFrame() {
  const w = canvas.width;
  const h = canvas.height;
  const maxSide = 1024;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  const outW = Math.round(w * scale);
  const outH = Math.round(h * scale);
  if (snap.width !== outW || snap.height !== outH) {
    snap.width = outW;
    snap.height = outH;
  }
  drawMirrored(snapCtx, outW, outH);
  return snap.toDataURL("image/jpeg", 0.85).split(",")[1];
}

async function generateAI() {
  aiState = "generating";
  aiError = "";
  aiLastGen = performance.now();
  try {
    const imageB64 = captureFrame();
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${AI_MODEL}:generateContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                { text: aiPrompt() },
                { inline_data: { mime_type: "image/jpeg", data: imageB64 } },
              ],
            },
          ],
        }),
      }
    );
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json())?.error?.message || detail;
      } catch {}
      throw new Error(`${res.status}: ${detail.slice(0, 80)}`);
    }
    const data = await res.json();
    const parts = data.candidates?.[0]?.content?.parts || [];
    const imgPart = parts.find((p) => p.inlineData || p.inline_data);
    if (!imgPart) throw new Error("no image returned — try again");
    const inline = imgPart.inlineData || imgPart.inline_data;
    const img = new Image();
    img.src = `data:${inline.mimeType || inline.mime_type};base64,${inline.data}`;
    await img.decode();
    aiImage = img;
    aiState = "ready";
  } catch (err) {
    console.error("AI generation failed:", err);
    // If a refresh failed but we still have a previous result, keep showing
    // it and retry quietly on the next refresh cycle.
    if (aiImage) {
      aiState = "ready";
      return;
    }
    aiError = err.message;
    aiState = "error";
    // Let the error linger, then allow another attempt.
    setTimeout(() => {
      if (aiState === "error") aiState = "idle";
    }, 4000);
  }
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

    // AI effect: generate once the frame is held steady, then keep
    // refreshing periodically so the stylized layer tracks the person.
    if (effect === "ai" && apiKey && aiState !== "generating" && aiState !== "error") {
      if (lastRawQuad) {
        const moved =
          targetQuad.reduce((s, p, i) => s + dist(p, lastRawQuad[i]), 0) / 4;
        stableFrames = moved < canvas.width * 0.01 ? stableFrames + 1 : 0;
      }
      const needFirst = !aiImage && stableFrames > 20;
      const needRefresh =
        aiImage &&
        stableFrames > 5 &&
        performance.now() - aiLastGen > AI_REFRESH_MS;
      if (needFirst || needRefresh) {
        stableFrames = 0;
        generateAI();
      }
    }
    lastRawQuad = targetQuad;
  } else if (corners && ++lostFrames <= MAX_LOST_FRAMES) {
    // Brief tracking dropout: hold the last quad instead of fading.
    presence = Math.min(1, presence + 0.12);
  } else {
    presence = Math.max(0, presence - 0.05);
    if (presence === 0) {
      corners = null;
      frameActive = false;
      // Frame fully dropped: clear the AI result so re-framing regenerates.
      aiImage = null;
      if (aiState === "ready" || aiState === "error") aiState = "idle";
      stableFrames = 0;
      lastRawQuad = null;
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
