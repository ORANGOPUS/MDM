"""
Train the Mycel Disposition Model and write an open-weight release:

  weights/model.safetensors   the weights
  config.json                 architecture + feature/label spec + tone anchors

Run:  python -m mycel_disposition_model.train
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .data import ANCHORS, sample
from .model import (
    FEATURES,
    MOODS,
    N_FEATURES,
    N_MOODS,
    N_TONE,
    TONE_AXES,
    MycelDispositionModel,
)

HERE = os.path.dirname(__file__)


def evaluate(model, f, tone, mood):
    model.eval()
    with torch.no_grad():
        logits, pred_tone, _ = model(f)
        acc = (logits.argmax(1) == mood).float().mean().item()
        tone_mae = (pred_tone - tone).abs().mean().item()
    return acc, tone_mae


def main():
    torch.manual_seed(0)
    hidden = 32

    f_tr, tone_tr, mood_tr = sample(12000, seed=1)
    f_va, tone_va, mood_va = sample(3000, seed=2)

    f_tr, tone_tr, mood_tr = map(torch.from_numpy, (f_tr, tone_tr, mood_tr))
    f_va, tone_va, mood_va = map(torch.from_numpy, (f_va, tone_va, mood_va))

    model = MycelDispositionModel(hidden=hidden)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

    epochs, batch = 240, 512
    n = f_tr.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            logits, pred_tone, _ = model(f_tr[idx])
            loss = F.cross_entropy(logits, mood_tr[idx]) + 2.0 * F.mse_loss(
                pred_tone, tone_tr[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            acc, mae = evaluate(model, f_va, tone_va, mood_va)
            print(f"epoch {epoch + 1:3d}  val mood_acc {acc:.3f}  tone_mae {mae:.3f}")

    acc, mae = evaluate(model, f_va, tone_va, mood_va)
    print(f"\nfinal  val mood_acc {acc:.3f}  tone_mae {mae:.3f}  params {n_params}")

    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        os.path.join(HERE, "weights", "model.safetensors"),
    )

    config = {
        "model_type": "mycel_disposition_model",
        "architecture": "mlp",
        "hidden": hidden,
        "n_parameters": n_params,
        "inputs": {"features": FEATURES, "n_features": N_FEATURES},
        "outputs": {
            "mood": {"labels": MOODS, "n": N_MOODS},
            "tone": {"axes": TONE_AXES, "n": N_TONE, "range": [-1, 1]},
        },
        "tone_anchors": {m: ANCHORS[i].tolist() for i, m in enumerate(MOODS)},
        "val_mood_accuracy": round(acc, 4),
        "val_tone_mae": round(mae, 4),
        "license": "apache-2.0",
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    print("wrote weights/model.safetensors and config.json")


if __name__ == "__main__":
    main()
