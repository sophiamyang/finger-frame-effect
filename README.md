# Finger Frame

**Live demo: https://sophiamyang.github.io/finger-frame-effect/**

Hold up both hands and frame a box with your index fingers and thumbs — a live
effect is applied inside the quad your fingers form, like the camera-framing
gesture effect.

Built with [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(browser WASM/GPU, loaded from CDN) + plain Canvas 2D. No build step.

## The finger-frame family

| App | Generation | Latency |
|---|---|---|
| **this app** — live camera, local effects | Canvas 2D (Van Gogh, toon, glitch, …) | none |
| [finger-frame-effect-ai](https://sophiamyang.github.io/finger-frame-effect-ai/) ([repo](https://github.com/sophiamyang/finger-frame-effect-ai)) — recorded video, AI restyle | Gemini Omni Flash (offline video edit) | minutes |
| [finger-frame-effect-lucy](https://sophiamyang.github.io/finger-frame-effect-lucy/) ([repo](https://github.com/sophiamyang/finger-frame-effect-lucy)) — live camera, live AI | Decart Lucy 2.5 (realtime video-to-video) | ~real time |
| [finger-pet](https://sophiamyang.github.io/finger-pet/) ([repo](https://github.com/sophiamyang/finger-pet)) — hand-tracked virtual pet cat | AI-generated sprites + game logic | none |

## Run

```bash
python3 -m http.server 8123
```

Then open http://localhost:8123 in a browser and allow camera access.
(Any static file server works; `getUserMedia` requires localhost or HTTPS.)

## How it works

- The webcam feed is drawn mirrored to a full-screen canvas.
- MediaPipe Hand Landmarker tracks up to 2 hands per frame (VIDEO mode, GPU).
- When both hands make an open "L" (thumb and index spread apart, scaled by
  hand size), the 4 tips (index + thumb of each hand) are sorted by angle
  around their centroid into a quad.
- Corners are exponentially smoothed and the effect fades in/out with a
  presence value, so the frame doesn't pop or jitter.
- The selected effect is drawn into the canvas clipped to the quad path, with
  a dashed outline + corner dots on top.

## Effects

Switch with the toolbar or keys 1–7:

1. **Pixelate** — mosaic censor effect (downscale + nearest-neighbor upscale)
2. **Blur** — frosted-glass blur
3. **Invert** — color negative
4. **Noir** — high-contrast black & white
5. **Glitch** — chromatic aberration, slice displacement, scanlines
6. **Toon** — cel-shaded cartoon version of the live feed: smoothed,
   posterized color bands + dark ink outlines from edge detection
7. **Van Gogh** — live painterly rendering (no AI): a smoothed orientation
   field is built from the image gradients each frame, then curved brush
   strokes follow it in two passes — a large brush for mass, a fine brush
   that sharpens edges — with per-stroke color jitter and slowly drifting
   swirl fields in flat areas. Moves with you in real time.

## Demo mode (no camera)

http://localhost:8123/?demo replaces the camera with a synthetic animated feed
and fake hand landmarks — useful for testing the rendering pipeline headlessly.
