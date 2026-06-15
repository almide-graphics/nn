"""T2-depth oracle: does a DIFFERENTIABLE circuit-DEPTH penalty move the
accuracy-vs-latency (depth) Pareto? This is the part Silicon Aware NN (area
only, Apr 2026) does NOT do.

Setup: a 2-layer DLGN. Layer 1 = H gate-neurons, each wiring two of the input
bits (fixed random wiring) through a learned 16-gate mix. Layer 2 = 1 output
neuron mixing two layer-1 outputs. Task: 3-bit parity (needs depth-2 of XORs).

DEPTH is made differentiable: each neuron's expected gate-depth =
  base_input_depth (max of its two parents' depths)  +  expected_gate_height,
where a TRIVIAL gate (wire/const: FALSE/A/B/TRUE/notA/notB) adds 0 (it passes a
parent through or is a constant of depth 0) and a real gate adds 1. The parent
"max" is relaxed with a softmax-weighted max so it's differentiable. Loss =
MSE + lambda_d * expected_output_depth. Sweep lambda_d; report (hardened
accuracy, hardened circuit depth).

    tools/.venv/bin/python research/autograd/oracle_depth.py
"""
import torch
torch.set_default_dtype(torch.float64)

TRIVIAL = {0, 3, 5, 15, 12, 10}            # FALSE,A,B,TRUE,notA,notB
HEIGHT = torch.tensor([0.0 if g in TRIVIAL else 1.0 for g in range(16)])  # gate adds 0 or 1
# which gate passes which parent (for hardened depth): trivial gates that are a
# pure wire of a parent inherit that parent's depth; const/real handled below.
# A (3) wires parent-a; B (5) wires parent-b; notA(12)->a; notB(10)->b (still
# depth+0 as "wire+inverter folds into routing"); FALSE/TRUE -> depth 0.


def relax_all(a, b):
    one = torch.ones_like(a); z = torch.zeros_like(a)
    return torch.stack([z, a*b, a-a*b, a, b-a*b, b, a+b-2*a*b, a+b-a*b,
                        one-(a+b-a*b), one-(a+b-2*a*b), one-b, one-b+a*b,
                        one-a, one-a+a*b, one-a*b, one], dim=-1)


# 2-bit XOR — solvable at depth 1 (one XOR gate), but the 2-layer net can also
# take a depth-2 detour (partial gate in L1, compose in L2). The depth penalty
# should push it toward the depth-1 solution: L1 computes XOR, L2 wires it
# through (a trivial pass-through gate, height 0). This is the redundancy a
# depth penalty can remove — the thing 3-bit parity (min depth 2) could not show.
X = torch.tensor([[float(i >> 1 & 1), float(i & 1)] for i in range(4)])
T = torch.tensor([float((i >> 1 & 1) ^ (i & 1)) for i in range(4)])

H = 2  # layer-1 neurons, both read the two input bits
W1 = [(0, 1), (0, 1)]
W2 = (0, 1)  # output neuron reads both L1 neurons


def soft_depth_parent(depths, weights):
    # differentiable "max" of parent depths, softmax-weighted (sharp)
    w = torch.softmax(depths * 8.0, dim=0)
    return (w * depths).sum()


def run(lam_d, steps=4000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    L1 = torch.randn(H, 16, requires_grad=True)
    L2 = torch.randn(1, 16, requires_grad=True)
    opt = torch.optim.Adam([L1, L2], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p1 = torch.softmax(L1, dim=1)               # (H,16)
        p2 = torch.softmax(L2, dim=1)               # (1,16)
        # layer 1 outputs
        outs1 = []
        for h in range(H):
            a = X[:, W1[h][0]]; b = X[:, W1[h][1]]
            R = relax_all(a, b)                      # (8,16)
            outs1.append(R @ p1[h])                  # (8,)
        o1 = torch.stack(outs1, dim=1)               # (8,H)
        # layer 2 output
        a = o1[:, W2[0]]; b = o1[:, W2[1]]
        R = relax_all(a, b)
        y = R @ p2[0]                                # (8,)
        mse = ((y - T) ** 2).mean()

        # differentiable expected depth
        d_in = torch.zeros(H)                         # input bits at depth 0
        exp_h1 = (p1 * HEIGHT).sum(dim=1)             # (H,) expected gate height
        d1 = exp_h1                                   # parents are inputs (depth 0)
        d2_parent = soft_depth_parent(torch.stack([d1[W2[0]], d1[W2[1]]]),
                                      None)
        exp_h2 = (p2[0] * HEIGHT).sum()
        depth = d2_parent + exp_h2
        loss = mse + lam_d * depth
        loss.backward()
        opt.step()

    with torch.no_grad():
        p1 = torch.softmax(L1, dim=1); p2 = torch.softmax(L2, dim=1)
        pick1 = [int(p1[h].argmax()) for h in range(H)]
        pick2 = int(p2[0].argmax())

        # hardened forward
        o1 = []
        for h in range(H):
            a = X[:, W1[h][0]]; b = X[:, W1[h][1]]
            o1.append(relax_all(a, b)[:, pick1[h]])
        o1 = torch.stack(o1, dim=1)
        yv = relax_all(o1[:, W2[0]], o1[:, W2[1]])[:, pick2]
        acc = ((yv > 0.5).double() == T).double().mean().item()

        # hardened depth: trivial gate adds 0, real gate adds 1; parent max
        def gh(g): return 0 if g in TRIVIAL else 1
        d1h = [gh(pick1[h]) for h in range(H)]        # parents = inputs depth 0
        depth_h = max(d1h[W2[0]], d1h[W2[1]]) + gh(pick2)
    return acc, depth_h, pick1, pick2


print(f"{'lam_d':>7} {'acc':>5} {'depth':>6}  picks (best-of-8)")
for lm in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
    best = None
    for s in range(8):
        acc, d, p1, p2 = run(lm, seed=s)
        key = (acc, -d)
        if best is None or key > best[0]:
            best = (key, acc, d, p1, p2)
    _, acc, d, p1, p2 = best
    print(f"{lm:>7.2f} {acc:>5.2f} {d:>6d}  L1={p1} L2={p2}")
