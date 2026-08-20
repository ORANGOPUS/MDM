"""
Load the trained Mycel Disposition Model and run it.

  python -m mycel_disposition_model.infer        # runs the blend demo
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file

from .model import FEATURES, MOODS, TONE_AXES, MycelDispositionModel

HERE = os.path.dirname(__file__)


def load(hidden: int = 32) -> MycelDispositionModel:
    model = MycelDispositionModel(hidden=hidden)
    state = load_file(os.path.join(HERE, "weights", "model.safetensors"))
    model.load_state_dict(state)
    model.eval()
    return model


def infer(model: MycelDispositionModel, **features):
    """features are keyword args named as in model.FEATURES; missing ones
    default to a neutral value (0.5 for levels, 0.0 for momenta)."""
    defaults = {name: (0.0 if name.startswith("d_") else 0.5) for name in FEATURES}
    defaults.update(features)
    x = torch.tensor([[defaults[name] for name in FEATURES]], dtype=torch.float32)
    with torch.no_grad():
        logits, tone, _ = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        tone = tone[0]
    mood = MOODS[int(probs.argmax())]
    return {
        "mood": mood,
        "confidence": round(float(probs.max()), 3),
        "tone": {axis: round(float(v), 3) for axis, v in zip(TONE_AXES, tone)},
    }


def _demo():
    model = load()
    print("Same dim, sparse environment. Only the user's interaction changes.\n")

    env = dict(density=0.2, luminosity=0.2, activity=0.2)

    withdrawn_user = infer(model, **env, tempo=0.3, valence=0.3, engagement=0.2)
    print("  quiet, low-engagement user ->", withdrawn_user)

    warm_user = infer(model, **env, tempo=0.5, valence=0.9, engagement=0.9)
    print("  warm, highly engaged user  ->", warm_user)

    frantic_user = infer(model, **env, tempo=0.95, valence=0.6, engagement=0.9)
    print("  fast, urgent user          ->", frantic_user)

    print(
        "\nThe environment never moved. Pure Synth.mood would return the same mood\n"
        "for all three; the learned model lets the person in the room shift Mycel."
    )


if __name__ == "__main__":
    _demo()
