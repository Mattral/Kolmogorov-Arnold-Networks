#!/usr/bin/env python
"""Minimal KAN torch training test."""
import os
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

import torch
from kanx.torch import KAN, Trainer
import numpy as np

print("Creating KAN model...")
model = KAN([2, 32, 1])
print(f"Model: {model}")

print("Creating training data...")
X = np.random.randn(128, 2).astype(np.float32)
y = np.sin(np.pi * X[:, :1]).astype(np.float32)

print("Training...")
hist = model.fit(X, y, epochs=3, batch_size=32, lr=1e-2, verbose=1)
print(f"Loss history: {hist.loss}")

print("Done!")
