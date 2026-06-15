#!/usr/bin/env python3
"""Tensor-autograd oracle: a tiny MLP forward+backward, fixed deterministic
inputs/weights, dumps loss and every parameter gradient for the Almide
tensor-autograd to match (~1e-5).

    MLP: x(2) -> W1(2x3)+b1 -> relu -> W2(3x2)+b2 -> softmax -> CE(target)

    tools/.venv/bin/python research/autograd/oracle_mlp.py
"""
import torch
torch.set_default_dtype(torch.float64)  # match Almide f64

# deterministic small tensors (row-major, same layout the Almide side builds)
x = torch.tensor([[0.5, -1.0]], requires_grad=False)          # (1,2)
W1 = torch.tensor([[0.1, 0.2, -0.3],
                   [0.4, -0.5, 0.6]], requires_grad=True)      # (2,3)
b1 = torch.tensor([[0.01, -0.02, 0.03]], requires_grad=True)  # (1,3)
W2 = torch.tensor([[0.7, -0.8],
                   [0.9, 0.1],
                   [-0.2, 0.5]], requires_grad=True)           # (3,2)
b2 = torch.tensor([[0.05, -0.06]], requires_grad=True)         # (1,2)
target = 1  # class index

h = x @ W1 + b1            # (1,3)
a = torch.relu(h)          # (1,3)
z = a @ W2 + b2            # (1,2)
logp = torch.log_softmax(z, dim=1)
loss = -logp[0, target]

loss.backward()

def dump(name, t):
    flat = t.detach().reshape(-1).tolist()
    print(f"{name} = " + ",".join(f"{v:.8f}" for v in flat))

print(f"loss = {loss.item():.8f}")
dump("dW1", W1.grad)
dump("db1", b1.grad)
dump("dW2", W2.grad)
dump("db2", b2.grad)
