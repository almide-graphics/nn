#!/usr/bin/env python3
"""T2 oracle: does a DIFFERENTIABLE gate-count penalty move the accuracy vs
circuit-size tradeoff (Pareto)? A 1-layer DLGN of N gate-neurons feeding a
fixed OR-readout learns 2-bit XOR; loss = MSE + lambda * expected_gate_count,
where trivial gates (wire/constant: FALSE,A,B,TRUE,notA,notB) cost 0 and real
gates cost 1. Sweep lambda; report (accuracy, expected real-gate count). If
raising lambda shrinks the circuit while holding accuracy, T2 has signal.

    tools/.venv/bin/python research/autograd/oracle_cost.py
"""
import torch
torch.set_default_dtype(torch.float64)

# cost per gate: 0 for trivial (wire/const), 1 for a real 2-input gate
# gate ids: 0 FALSE,1 AND,2 A&!B,3 A,4 !A&B,5 B,6 XOR,7 OR,8 NOR,9 XNOR,
#           10 !B,11 A|!B,12 !A,13 !A|B,14 NAND,15 TRUE
TRIVIAL = {0, 3, 5, 15, 12, 10}  # FALSE, A, B, TRUE, notA, notB
COST = torch.tensor([0.0 if g in TRIVIAL else 1.0 for g in range(16)])


def relax_all(a, b):
    one = torch.ones_like(a)
    z = torch.zeros_like(a)
    return torch.stack([
        z, a * b, a - a * b, a, b - a * b, b,
        a + b - 2 * a * b, a + b - a * b,
        one - (a + b - a * b), one - (a + b - 2 * a * b),
        one - b, one - b + a * b, one - a, one - a + a * b,
        one - a * b, one,
    ], dim=-1)  # (..., 16)


# 4 XOR examples
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T = torch.tensor([0.0, 1.0, 1.0, 0.0])
N = 4  # gate-neurons in the layer; each sees inputs (a,b)


def train(lmbda, steps=3000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    # random init breaks the inter-neuron symmetry — otherwise all neurons
    # move together (all-XOR at lambda=0, all-FALSE at lambda>0) and never
    # find the asymmetric optimum "1 XOR + 3 FALSE".
    logits = (torch.randn(N, 16) * 1.0).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    # fixed OR readout: out = 1 - prod(1 - neuron_i). With OR, the layer can
    # solve XOR with ONE real gate (a XOR-neuron) + the rest FALSE (trivial,
    # cost 0), so there is genuine redundancy for the cost penalty to remove.
    Rn = relax_all(X[:, 0], X[:, 1])                   # (4,16) const
    for _ in range(steps):
        opt.zero_grad()
        p = torch.softmax(logits, dim=1)              # (N,16)
        outs = Rn @ p.T                                # (4,N) each neuron's mix
        y = 1.0 - torch.prod(1.0 - outs, dim=1)        # (4,) soft-OR readout
        mse = ((y - T) ** 2).mean()
        exp_cost = (p * COST).sum(dim=1).sum()         # expected real-gate count
        loss = mse + lmbda * exp_cost
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.softmax(logits, dim=1)
        pick = p.argmax(dim=1)                          # hardened gate per neuron
        real_gates = sum(0 if int(g) in TRIVIAL else 1 for g in pick)
        hard_neurons = torch.stack([Rn[:, g] for g in pick], dim=1)  # (4,N), 0/1
        hard_out = 1.0 - torch.prod(1.0 - hard_neurons, dim=1)       # OR
        acc = ((hard_out > 0.5).double() == T).double().mean().item()
        exp_real = (p * COST).sum().item()
    return acc, real_gates, exp_real, [int(g) for g in pick]


# each lambda: best of several seeds (standard for a non-convex circuit search)
print(f"{'lambda':>8} {'hard_acc':>9} {'real_gates':>11}  picks (best-of-8 seeds)")
for lm in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]:
    best = None
    for s in range(8):
        acc, rg, er, picks = train(lm, seed=s)
        # prefer higher accuracy, then fewer real gates
        key = (acc, -rg)
        if best is None or key > best[0]:
            best = (key, acc, rg, picks)
    _, acc, rg, picks = best
    print(f"{lm:>8.2f} {acc:>9.2f} {rg:>11d}  {picks}")
