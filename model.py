"""
Mycel Disposition Model (MDM)

The neural half of Mycel's neurosymbolic layer. A small MLP that maps
environment state + interaction state to:

  - mood_logits : 6-way classification over Mycel's moods
  - tone        : 3 continuous disposition axes in [-1, 1]
                  (arousal, brightness, openness)

The 3-axis tone vector is the "disposition embedding" the mimetic layer
consumes to modulate response generation. Mood is the interpretable,
UI-facing label; tone is the continuous signal.

Tiny by design (~a few thousand parameters) so it trains in seconds on an
M1 and exports cleanly to ONNX for in-Biome (Node / browser) inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Canonical feature order. Inference MUST build the input vector in this order.
FEATURES = [
    "density",        # env: point-cloud density        [0, 1]
    "luminosity",     # env: ambient luminosity          [0, 1]
    "activity",       # env: motion / activity           [0, 1]
    "d_density",      # env: density momentum            [-1, 1]
    "d_luminosity",   # env: luminosity momentum         [-1, 1]
    "d_activity",     # env: activity momentum           [-1, 1]
    "tempo",          # interaction: message cadence     [0, 1]
    "valence",        # interaction: affective valence   [0, 1]
    "verbosity",      # interaction: message length      [0, 1]
    "engagement",     # interaction: attention / recency [0, 1]
]

MOODS = ["withdrawn", "charged", "luminous", "dense", "restless", "settled"]

TONE_AXES = ["arousal", "brightness", "openness"]

N_FEATURES = len(FEATURES)
N_MOODS = len(MOODS)
N_TONE = len(TONE_AXES)


class MycelDispositionModel(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mood_head = nn.Linear(hidden, N_MOODS)
        self.tone_head = nn.Linear(hidden, N_TONE)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mood_logits = self.mood_head(h)
        tone = torch.tanh(self.tone_head(h))
        return mood_logits, tone, h
