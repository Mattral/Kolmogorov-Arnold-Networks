#!/usr/bin/env python
"""Pure PyTorch KAN test without TensorFlow."""
import os
os.environ["TORCH_INDUCTOR_DISABLE_TRITON"] = "1"

import torch
# Don't import tensorflow at all
import numpy as np

print("PyTorch version:", torch.__version__)
print("Creating simple module...")

class SimpleKAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(2, 32)
        self.lin2 = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

model = SimpleKAN()
print(f"Model: {model}")

print("Creating optimizer...")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
print("Optimizer created OK")

print("Creating training data...")
X = torch.randn(128, 2)
y = torch.sin(torch.pi * X[:, :1])

print("Training...")
for epoch in range(3):
    out = model(X)
    loss = torch.nn.functional.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print("Done!")
