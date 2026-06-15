#!/usr/bin/env python3
"""Gradient oracle for ONE differentiable-logic-gate neuron (DLGN building
block). out_i = sum_g softmax(logits)_g * relax_g(a_i, b_i).

Verifies the Almide gate-neuron op's vjp to a, b, AND logits — i.e. that the
16 gate relaxations are correct differentiable ops with gradients flowing
through them. torch computes the reference grads via its own autograd.

    tools/.venv/bin/python research/autograd/oracle_gate.py
"""
import torch
torch.set_default_dtype(torch.float64)


def relax(g, a, b):
    one = torch.ones_like(a)
    return [
        torch.zeros_like(a),        # 0 FALSE
        a * b,                      # 1 AND
        a - a * b,                  # 2 A and not B
        a,                          # 3 A
        b - a * b,                  # 4 not A and B
        b,                          # 5 B
        a + b - 2 * a * b,          # 6 XOR
        a + b - a * b,              # 7 OR
        one - (a + b - a * b),      # 8 NOR
        one - (a + b - 2 * a * b),  # 9 XNOR
        one - b,                    # 10 not B
        one - b + a * b,            # 11 A or not B
        one - a,                    # 12 not A
        one - a + a * b,            # 13 not A or B
        one - a * b,                # 14 NAND
        one,                        # 15 TRUE
    ][g]


a = torch.tensor([0.3, 0.7], requires_grad=True)
b = torch.tensor([0.6, 0.2], requires_grad=True)
logits = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.4, -0.3,
                       0.2, 0.15, -0.25, 0.1, 0.05, -0.05, 0.35, -0.4],
                      requires_grad=True)

p = torch.softmax(logits, dim=0)              # (16,)
R = torch.stack([relax(g, a, b) for g in range(16)], dim=1)  # (2,16)
out = R @ p                                   # (2,)
c = torch.tensor([1.0, -2.0])
loss = (out * c).sum()
loss.backward()


def dump(name, t):
    print(f"{name} = " + ",".join(f"{v:.8f}" for v in t.reshape(-1).tolist()))


dump("out", out)
dump("da", a.grad)
dump("db", b.grad)
dump("dlogits", logits.grad)
