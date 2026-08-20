# Mycel Disposition Model (MDM)

An open-weight model for **Mycel**, the AI companion in Orangopus' Biome. It is
the *neural half* of Mycel's neurosymbolic layer: a small MLP that maps the
Biome's environment state **and** the user's interaction state to Mycel's
disposition.

- **License:** Apache-2.0
- **Parameters:** 1,705
- **Format:** `safetensors` (+ ONNX for in-Biome inference)
- **Val mood accuracy:** 0.932 · **Val tone MAE:** 0.015
- **Trains in:** seconds, CPU — no GPU required (fine on an M1)

## What it does

| | |
|---|---|
| **Input** | 10 features: 3 environment levels + 3 environment momenta + 4 interaction signals |
| **Output** | `mood` (6-way label) and `tone` (3 continuous axes in [-1, 1]) |

Mood is the interpretable, on-screen label. Tone — `arousal`, `brightness`,
`openness` — is the continuous **disposition vector** the mimetic layer consumes
to modulate how Mycel responds.

### Feature order (canonical — build inputs in this order)

```
density  luminosity  activity  d_density  d_luminosity  d_activity
tempo  valence  verbosity  engagement
```

Environment levels and interaction signals are in `[0, 1]`; momenta (`d_*`) are
in `[-1, 1]`.

## Why it exists

Mycel's original mood logic (`Synth.mood` in the Fable port) was pure-environment
thresholds — the same dim, sparse room always produced the same mood, no matter
who was in it. MDM keeps that behaviour when interaction is neutral, but it
**generalises the hard thresholds into smooth gradients** and lets interaction
state pull the disposition. That env × interaction blend is the seam the rule-based
version couldn't express cleanly.

The sub-100% mood accuracy is intentional: misses cluster at anchor boundaries
where the network interpolates between two adjacent moods instead of snapping —
which is the threshold-softening the design wants, not an error.

### The blend, demonstrated

Same dim, sparse environment (`density=luminosity=activity=0.2`); only the user
changes:

```
quiet, low-engagement user  -> withdrawn  (openness -0.42)
warm, highly engaged user   -> settled    (openness +0.51)
fast, urgent user           -> dense      (arousal raised)
```

## How it was trained

No real user data. Training targets come from a defined **tone geometry** (see
`data.py`): six mood anchors in `(arousal, brightness, openness)` space, a
hand-authored map from features to a continuous tone vector, and mood as the
nearest anchor. The model learns both heads jointly (cross-entropy on mood +
MSE on tone). Because targets are synthetic and rule-derived, the release carries
no privacy surface — but it is a faithful, differentiable stand-in you can later
fine-tune on real interaction logs.

## Files

```
mycel_disposition_model/
  model.py          nn.Module + canonical feature / label spec
  data.py           tone geometry + synthetic data generator
  train.py          trains, writes weights/model.safetensors + config.json
  infer.py          load() + infer() + the blend demo
  export_onnx.py    weights/mycel-disposition.onnx for in-Biome inference
  disposition.js    onnxruntime helper for Node / browser (the Biome)
  config.json       architecture + spec + tone anchors + metrics
  weights/
    model.safetensors
    mycel-disposition.onnx (+ .onnx.data)
```

## Usage

Python:

```bash
pip install -r requirements.txt
python -m mycel_disposition_model.train        # reproduce the weights
python -m mycel_disposition_model.infer         # run the blend demo
```

In the Biome (Node / browser), via `disposition.js`:

```js
import * as ort from 'onnxruntime-web';
import { loadDisposition } from './disposition.js';

const infer = await loadDisposition(ort, '/models/mycel-disposition.onnx');
const { mood, tone } = await infer({
  density: biome.density, luminosity: biome.luminosity, activity: biome.activity,
  tempo: user.tempo, valence: user.valence, engagement: user.engagement,
});
mycel.setDisposition(tone);   // continuous — drives tone
ui.setMoodTag(mood);          // discrete — the label
```

## Limitations

- It models **disposition**, not language. It decides Mycel's mood/tone; it does
  not generate text. Pair it with your existing dialogue path.
- Interaction features (`tempo`, `valence`, `verbosity`, `engagement`) are
  assumed pre-computed upstream. Garbage in, garbage out — a good valence
  estimator matters more than this model's size.
- Trained on synthetic targets, so it encodes *your stated aesthetic*, not
  empirical user reactions. Fine-tune on logged interactions to close that gap.
- Tiny by design. If you later want per-user adaptation or richer tone geometry,
  widen `hidden` and add axes — the training loop scales without changes.
