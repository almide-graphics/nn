#!/usr/bin/env python3
"""Gradient oracle for the Almide scalar-autograd parity check.

Same expression as research/autograd/grad.almd, computed with PyTorch's
autograd. The Almide tape-autograd must match value + grads to ~1e-6.

    tools/.venv/bin/python research/autograd/oracle.py
"""
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)

c = a * b          # mul
d = c + a          # add (a is REUSED → grad must accumulate)
e = torch.tanh(d)  # tanh
L = e

L.backward()

print(f"value L = {L.item():.10f}")
print(f"dL/da   = {a.grad.item():.10f}")
print(f"dL/db   = {b.grad.item():.10f}")
