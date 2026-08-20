"""
Synthetic training data for the Mycel Disposition Model.

This is where the neurosymbolic bridge happens. Mycel's original mood logic
(Synth.mood in the Fable port) was pure-environment thresholds. Here we define
a continuous *tone geometry* driven by BOTH environment and interaction state,
then derive the mood label as the nearest mood anchor in that tone space.

The model trained on this data reproduces the original rules when interaction
is neutral, but generalises the hard thresholds into smooth gradients and, more
importantly, lets interaction state pull the disposition -- the env x interaction
blend that couldn't be written cleanly as if/else.

No real user data is used. Everything is generated from the definitions below,
so the release carries no privacy surface.
"""

from __future__ import annotations

import numpy as np

from .model import MOODS

# Prototypical (arousal, brightness, openness) for each mood.
# Mood = nearest anchor to the continuous tone vector.
ANCHORS = np.array(
    [
        [-0.60, -0.60, -0.70],  # withdrawn
        [0.80, 0.20, 0.40],     # charged
        [-0.20, 0.80, 0.50],    # luminous
        [0.10, -0.60, -0.20],   # dense
        [0.70, 0.60, 0.20],     # restless
        [-0.30, 0.10, 0.10],    # settled
    ],
    dtype=np.float32,
)
assert ANCHORS.shape[0] == len(MOODS)


def _clip(v):
    return np.clip(v, -1.0, 1.0)


def tone_from_features(f: np.ndarray) -> np.ndarray:
    """Map a feature matrix (N, 10) to a tone matrix (N, 3).

    This is the hand-authored target geometry the network learns to imitate.
    Column order matches model.FEATURES.
    """
    den, lum, act = f[:, 0], f[:, 1], f[:, 2]
    dact = f[:, 5]
    tempo, valence, verbosity, engage = f[:, 6], f[:, 7], f[:, 8], f[:, 9]

    # arousal: driven by activity + how fast the user is moving, nudged by
    # upward activity momentum and long messages.
    arousal = _clip(
        (0.50 * act + 0.32 * tempo + 0.10 * np.maximum(dact, 0.0) + 0.08 * verbosity) * 2.0 - 1.0
    )

    # brightness: mostly ambient light, warmed slightly by positive valence.
    brightness = _clip((0.82 * lum + 0.18 * valence) * 2.0 - 1.0)

    # openness: mostly interaction (engagement, valence), dimmed by a dark,
    # crowded environment.
    openness = _clip(
        (0.42 * engage + 0.30 * valence + 0.16 * lum + 0.12 * (1.0 - den)) * 2.0 - 1.0
    )

    return np.stack([arousal, brightness, openness], axis=1).astype(np.float32)


def mood_from_tone(tone: np.ndarray) -> np.ndarray:
    """Nearest mood anchor for each tone vector -> integer labels (N,)."""
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3) -> (N, M) distances
    d = np.linalg.norm(tone[:, None, :] - ANCHORS[None, :, :], axis=2)
    return d.argmin(axis=1).astype(np.int64)


def sample(n: int, seed: int = 0):
    """Generate (features, tone, mood) with a mild curriculum:

    ~70% of samples have neutral-ish interaction (so the model firmly learns
    the environment rules), ~30% have wide-ranging interaction (so it learns
    the blend). Returns float32 features/tone and int64 mood labels.
    """
    rng = np.random.default_rng(seed)

    # Environment levels in [0, 1], momenta in [-1, 1].
    den = rng.random(n)
    lum = rng.random(n)
    act = rng.random(n)
    dden = rng.uniform(-1, 1, n)
    dlum = rng.uniform(-1, 1, n)
    dact = rng.uniform(-1, 1, n)

    # Interaction: 70% clustered near neutral (0.5), 30% uniform.
    def inter():
        neutral = rng.normal(0.5, 0.12, n)
        wide = rng.random(n)
        mask = rng.random(n) < 0.70
        return np.clip(np.where(mask, neutral, wide), 0.0, 1.0)

    tempo, valence, verbosity, engage = inter(), inter(), inter(), inter()

    f = np.stack(
        [den, lum, act, dden, dlum, dact, tempo, valence, verbosity, engage], axis=1
    ).astype(np.float32)

    tone = tone_from_features(f)
    # A little label noise on the tone before snapping to a mood keeps the
    # classification boundaries from being razor-sharp.
    noisy = _clip(tone + rng.normal(0, 0.04, tone.shape).astype(np.float32))
    mood = mood_from_tone(noisy)

    return f, tone, mood
