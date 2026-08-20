// disposition.js
//
// In-Biome inference for the Mycel Disposition Model.
// Runs the ONNX weights via onnxruntime — works in Node (onnxruntime-node)
// and the browser (onnxruntime-web). This is the neural half of Mycel's
// mimetic layer: feed it environment + interaction state, get back a mood
// and a continuous tone vector to modulate response generation.
//
//   npm i onnxruntime-web        # browser / Biome front-end
//   npm i onnxruntime-node       # server-side
//
// import * as ort from 'onnxruntime-web';   // or 'onnxruntime-node'

// Canonical feature order — must match model.FEATURES exactly.
export const FEATURES = [
  'density', 'luminosity', 'activity',
  'd_density', 'd_luminosity', 'd_activity',
  'tempo', 'valence', 'verbosity', 'engagement',
];

export const MOODS = ['withdrawn', 'charged', 'luminous', 'dense', 'restless', 'settled'];
export const TONE_AXES = ['arousal', 'brightness', 'openness'];

const DEFAULTS = Object.fromEntries(
  FEATURES.map((n) => [n, n.startsWith('d_') ? 0.0 : 0.5]),
);

function softmax(a) {
  const m = Math.max(...a);
  const e = a.map((x) => Math.exp(x - m));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map((x) => x / s);
}

export async function loadDisposition(ort, url = './mycel-disposition.onnx') {
  const session = await ort.InferenceSession.create(url);

  return async function infer(state = {}) {
    const merged = { ...DEFAULTS, ...state };
    const vec = Float32Array.from(FEATURES.map((n) => merged[n]));
    const input = new ort.Tensor('float32', vec, [1, FEATURES.length]);

    const out = await session.run({ features: input });
    const logits = Array.from(out.mood_logits.data);
    const tone = Array.from(out.tone.data);
    const probs = softmax(logits);

    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

    return {
      mood: MOODS[best],
      confidence: +probs[best].toFixed(3),
      tone: Object.fromEntries(TONE_AXES.map((ax, i) => [ax, +tone[i].toFixed(3)])),
      // hidden trunk activation is also available as out.hidden.data if the
      // mimetic layer wants the raw embedding instead of the 3 tone axes.
    };
  };
}

// Example wiring into the Biome tick loop:
//
//   const infer = await loadDisposition(ort, '/models/mycel-disposition.onnx');
//   const { mood, tone } = await infer({
//     density: biome.density, luminosity: biome.luminosity, activity: biome.activity,
//     tempo: user.tempo, valence: user.valence, engagement: user.engagement,
//   });
//   mycel.setDisposition(tone);   // continuous — drives response tone
//   ui.setMoodTag(mood);          // discrete — the label on screen
