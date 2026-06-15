"""The payoff probe: on a REAL multi-layer DLGN circuit (the prerequisite the
depth probe demanded), does a gate-count penalty move the accuracy-vs-circuit-
size Pareto? Toy XOR showed it for 1 layer; here we test it on the deep,
non-trivial circuit ((x0^x1)|(x2&x3)) where redundancy genuinely exists.

Same net as oracle_deep.py (2 hidden layers x WIDTH, random wiring) + a
differentiable expected-gate-count penalty (trivial wire/const gates cost 0,
real gates cost 1). Sweep lambda; report hardened accuracy + real-gate count.
If raising lambda shrinks the circuit while holding 16/16, T2 holds at depth.

    tools/.venv/bin/python research/autograd/oracle_deep_cost.py
"""
import torch
torch.set_default_dtype(torch.float64)

TRIVIAL = {0, 3, 5, 15, 12, 10}
COST = torch.tensor([0.0 if g in TRIVIAL else 1.0 for g in range(16)])


def relax_all(a, b):
    one = torch.ones_like(a); z = torch.zeros_like(a)
    return torch.stack([z, a*b, a-a*b, a, b-a*b, b, a+b-2*a*b, a+b-a*b,
                        one-(a+b-a*b), one-(a+b-2*a*b), one-b, one-b+a*b,
                        one-a, one-a+a*b, one-a*b, one], dim=-1)


NBITS = 4
X = torch.tensor([[float(i >> (NBITS-1-k) & 1) for k in range(NBITS)] for i in range(16)])
T = torch.tensor([float(((int(x[0]) ^ int(x[1])) | (int(x[2]) & int(x[3])))) for x in X])
WIDTH, DEPTH = 6, 2


def build_wiring(seed):
    g = torch.Generator().manual_seed(seed)
    wir = []; prev = NBITS
    for _ in range(DEPTH):
        wir.append([(int(torch.randint(prev,(1,),generator=g)),
                     int(torch.randint(prev,(1,),generator=g))) for _ in range(WIDTH)])
        prev = WIDTH
    readout = (int(torch.randint(prev,(1,),generator=g)), int(torch.randint(prev,(1,),generator=g)))
    return wir, readout


def run(lmbda, seed, steps=5000, lr=0.05):
    torch.manual_seed(seed)
    wiring, readout = build_wiring(seed)
    params = [torch.randn(WIDTH,16,requires_grad=True) for _ in range(DEPTH)]
    rdo = torch.randn(16, requires_grad=True)
    opt = torch.optim.Adam(params + [rdo], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        cur = [X[:,k] for k in range(NBITS)]
        cost = 0.0
        for li in range(DEPTH):
            nxt = []
            for h in range(WIDTH):
                ia, ib = wiring[li][h]
                p = torch.softmax(params[li][h], dim=0)
                nxt.append(relax_all(cur[ia], cur[ib]) @ p)
                cost = cost + (p * COST).sum()
            cur = nxt
        ia, ib = readout
        pr = torch.softmax(rdo, dim=0)
        y = relax_all(cur[ia], cur[ib]) @ pr
        cost = cost + (pr * COST).sum()
        loss = ((y - T)**2).mean() + lmbda * cost
        loss.backward(); opt.step()
    with torch.no_grad():
        cur = [X[:,k] for k in range(NBITS)]; real = 0
        for li in range(DEPTH):
            nxt = []
            for h in range(WIDTH):
                ia, ib = wiring[li][h]
                g = int(params[li][h].argmax()); real += 0 if g in TRIVIAL else 1
                nxt.append(relax_all(cur[ia], cur[ib])[:, g])
            cur = nxt
        ia, ib = readout
        g = int(rdo.argmax()); real += 0 if g in TRIVIAL else 1
        yh = relax_all(cur[ia], cur[ib])[:, g]
        acc = ((yh > 0.5).double() == T).double().mean().item()
    return acc, real


print(f"deep DLGN ({DEPTH}x{WIDTH}) on ((x0^x1)|(x2&x3)); gate-count penalty sweep")
print(f"{'lambda':>8} {'acc':>5} {'real_gates':>11}  (best-of-12 seeds: max acc, then min gates)")
for lm in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
    best = None
    for s in range(12):
        acc, real = run(lm, s)
        key = (acc, -real)
        if best is None or key > best[0]:
            best = (key, acc, real)
    _, acc, real = best
    print(f"{lm:>8.3f} {acc:>5.2f} {real:>11d}")
