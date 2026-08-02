# Finger Frame

Hold up both hands and frame a box with your index fingers and thumbs — a live
effect is applied inside the quad your fingers form, like the camera-framing
gesture effect.

Built with [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
(browser WASM/GPU, loaded from CDN) + plain Canvas 2D. No build step.

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

Switch with the toolbar or keys 1–6:

1. **Pixelate** — mosaic censor effect (downscale + nearest-neighbor upscale)
2. **Blur** — frosted-glass blur
3. **Invert** — color negative
4. **Noir** — high-contrast black & white
5. **Glitch** — chromatic aberration, slice displacement, scanlines
6. **Toon** — cel-shaded cartoon version of the live feed: smoothed,
   posterized color bands + dark ink outlines from edge detection
7. **AI ✨** — hold the frame steady for ~1 second and Gemini's image model
   redraws the framed region as a 3D animated movie character, warped back
   into your finger frame (requires an API key, see below)

## AI effect setup (bring your own key)

The AI effect uses Google's Gemini image model (`gemini-2.5-flash-image`)
via image-to-image editing. Click the 🔑 button, paste a key from
[Google AI Studio](https://aistudio.google.com/apikey), and select the
AI ✨ effect.

- Your key is stored only in your browser (localStorage if "remember" is
  checked, sessionStorage otherwise) and sent only to
  `generativelanguage.googleapis.com`.
- Use a dedicated free-tier key without billing attached; optionally
  restrict it by HTTP referrer in Google Cloud console.
- Each generation is one image request (~$0.04 on paid tier, or free-tier
  quota). A new generation triggers each time you re-form the frame after
  fully dropping it.

## Demo mode (no camera)

http://localhost:8123/?demo replaces the camera with a synthetic animated feed
and fake hand landmarks — useful for testing the rendering pipeline headlessly.
