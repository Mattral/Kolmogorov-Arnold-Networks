#!/usr/bin/env python
"""Test PyTorch with TensorFlow imported."""
import os
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

print("Importing TensorFlow...")
import tensorflow as tf
print("TensorFlow OK")

print("Importing PyTorch...")
import torch
import numpy as np
print("PyTorch OK")

print("Creating simple torch model...")
model = torch.nn.Sequential(
    torch.nn.Linear(2, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
print("Model created")

print("Creating optimizer...")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
print("Optimizer OK")

print("Training...")
X = torch.randn(128, 2)
y = torch.sin(torch.pi * X[:, :1])

for epoch in range(3):
    out = model(X)
    loss = torch.nn.functional.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print("Done!")
