"""L_depth at real scale: on a multi-layer DLGN with genuine DEPTH redundancy,
does a differentiable circuit-DEPTH penalty move the accuracy-vs-latency
(depth) Pareto? This is the part Silicon Aware NN (area only, image scale)
does NOT do — depth/latency in the loss, on a real multi-stage circuit.

Key vs the failed toy probe: pick a task that is solvable SHALLOW but whose
deep net can also take a deeper detour, so depth has slack to remove. Task:
f(x) = x0 XOR x1  (depth-1 function), but the network has DEPTH hidden layers
feeding an output — so the net can solve it at depth 1 (compute XOR early,
wire it up = trivial pass-through gates, height 0) OR waste depth. The depth
penalty should push toward the shallow realization.

Differentiable depth: each neuron's expected depth =
   softmax-weighted-max(parent depths)  +  sum_g p_g * height_g
(trivial wire/const gate adds 0 height; real gate adds 1). Loss = MSE +
lambda_d * output_depth. Sweep lambda_d; report hardened accuracy + hardened
depth (real critical path).

    tools/.venv/bin/python research/autograd/oracle_deep_depth.py
"""
import torch
torch.set_default_dtype(torch.float64)

TRIVIAL = {0, 3, 5, 15, 12, 10}                 # wire/const: height 0
HEIGHT = torch.tensor([0.0 if g in TRIVIAL else 1.0 for g in range(16)])
# which parent a trivial WIRE gate passes (for hardened depth bookkeeping):
# A(3)->parent a, B(5)->parent b, notA(12)->a, notB(10)->b; FALSE/TRUE -> depth 0


def relax_all(a, b):
    one = torch.ones_like(a); z = torch.zeros_like(a)
    return torch.stack([z, a*b, a-a*b, a, b-a*b, b, a+b-2*a*b, a+b-a*b,
                        one-(a+b-a*b), one-(a+b-2*a*b), one-b, one-b+a*b,
                        one-a, one-a+a*b, one-a*b, one], dim=-1)


NBITS = 4
X = torch.tensor([[float(i >> (NBITS-1-k) & 1) for k in range(NBITS)] for i in range(16)])
T = torch.tensor([float(int(x[0]) ^ int(x[1])) for x in X])   # XOR of first 2 bits
WIDTH, DEPTH = 4, 3        # 3 hidden layers: depth slack exists (XOR needs 1)


def build_wiring(seed):
    g = torch.Generator().manual_seed(seed)
    wir = []; prev = NBITS
    for _ in range(DEPTH):
        wir.append([(int(torch.randint(prev,(1,),generator=g)),
                     int(torch.randint(prev,(1,),generator=g))) for _ in range(WIDTH)])
        prev = WIDTH
    readout = (int(torch.randint(prev,(1,),generator=g)), int(torch.randint(prev,(1,),generator=g)))
    return wir, readout


def soft_max(vals):
    # differentiable max via softmax-weighting (sharp)
    v = torch.stack(vals)
    w = torch.softmax(v * 8.0, dim=0)
    return (w * v).sum()


def run(lam_d, seed, steps=3000, lr=0.05):
    torch.manual_seed(seed)
    wiring, readout = build_wiring(seed)
    params = [torch.randn(WIDTH, 16, requires_grad=True) for _ in range(DEPTH)]
    rdo = torch.randn(16, requires_grad=True)
    opt = torch.optim.Adam(params + [rdo], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        cur = [X[:, k] for k in range(NBITS)]
        depths = [torch.zeros(()) for _ in range(NBITS)]   # input depth 0
        for li in range(DEPTH):
            nxt, ndep = [], []
            for h in range(WIDTH):
                ia, ib = wiring[li][h]
                p = torch.softmax(params[li][h], dim=0)
                nxt.append(relax_all(cur[ia], cur[ib]) @ p)
                eh = (p * HEIGHT).sum()
                ndep.append(soft_max([depths[ia], depths[ib]]) + eh)
            cur, depths = nxt, ndep
        ia, ib = readout
        pr = torch.softmax(rdo, dim=0)
        y = relax_all(cur[ia], cur[ib]) @ pr
        out_depth = soft_max([depths[ia], depths[ib]]) + (pr * HEIGHT).sum()
        loss = ((y - T)**2).mean() + lam_d * out_depth
        loss.backward(); opt.step()

    with torch.no_grad():
        cur = [X[:, k] for k in range(NBITS)]
        hdep = [0 for _ in range(NBITS)]
        for li in range(DEPTH):
            nxt, ndep = [], []
            for h in range(WIDTH):
                ia, ib = wiring[li][h]
                g = int(params[li][h].argmax())
                nxt.append(relax_all(cur[ia], cur[ib])[:, g])
                # hardened depth: trivial wire inherits a parent's depth (+0);
                # real gate = max(parents)+1; const (FALSE/TRUE) = 0
                if g in (0, 15):
                    ndep.append(0)
                elif g in (3, 12):       # wires parent a
                    ndep.append(hdep[ia])
                elif g in (5, 10):       # wires parent b
                    ndep.append(hdep[ib])
                else:
                    ndep.append(max(hdep[ia], hdep[ib]) + 1)
            cur, hdep = nxt, ndep
        ia, ib = readout
        g = int(rdo.argmax())
        if g in (0, 15): od = 0
        elif g in (3, 12): od = hdep[ia]
        elif g in (5, 10): od = hdep[ib]
        else: od = max(hdep[ia], hdep[ib]) + 1
        yh = relax_all(cur[ia], cur[ib])[:, g]
        acc = ((yh > 0.5).double() == T).double().mean().item()
    return acc, od


print(f"deep DLGN ({DEPTH}L x{WIDTH}) on XOR(x0,x1) [depth-1 task, depth-{DEPTH} net]")
print(f"  -> does the depth penalty shallow the DISTRIBUTION of solved circuits?")
print(f"{'lam_d':>8} {'n_solved':>9} {'mean_depth':>11} {'depths(solved, sorted)':>26}")
import sys
for lm in [0.0, 0.01, 0.03, 0.06, 0.1, 0.2]:
    rows = [run(lm, s) for s in range(8)]
    solved = sorted(d for a, d in rows if a == 1.0)   # depths of 16/16 circuits
    md = (sum(solved) / len(solved)) if solved else float("nan")
    print(f"{lm:>8.3f} {len(solved):>9d} {md:>11.2f}   {solved}")
    sys.stdout.flush()
