"""Scaling probe: can a MULTI-LAYER DLGN learn a non-trivial multi-stage
boolean function (beyond XOR/parity) and harden to an exact circuit? This is
the prerequisite the depth-Pareto probe revealed — depth redundancy only
exists in real multi-layer circuits, so first prove deep DLGNs train + harden.

Task: 4-bit input, target f(x) = ((x0 XOR x1) OR (x2 AND x3)). Needs two logic
stages: a layer of {XOR, AND} then an OR — genuinely depth-2, non-trivial.

Architecture (difflogic-style): L layers of H gate-neurons; each neuron has a
fixed random wiring to two outputs of the previous layer; learned 16-gate
softmax mix; final readout = one extra neuron. Train on all 16 inputs (learn
the function exactly), MSE + Adam, then harden (argmax gate) and check 16/16.

    tools/.venv/bin/python research/autograd/oracle_deep.py
"""
import torch
torch.set_default_dtype(torch.float64)


def relax_all(a, b):
    one = torch.ones_like(a); z = torch.zeros_like(a)
    return torch.stack([z, a*b, a-a*b, a, b-a*b, b, a+b-2*a*b, a+b-a*b,
                        one-(a+b-a*b), one-(a+b-2*a*b), one-b, one-b+a*b,
                        one-a, one-a+a*b, one-a*b, one], dim=-1)


NBITS = 4
X = torch.tensor([[float(i >> (NBITS-1-k) & 1) for k in range(NBITS)]
                  for i in range(2**NBITS)])           # (16,4)
T = torch.tensor([float(((int(x[0]) ^ int(x[1])) | (int(x[2]) & int(x[3]))))
                  for x in X])                          # (16,)

WIDTH = 6     # neurons per hidden layer
DEPTH = 2     # hidden layers


def build_wiring(seed):
    g = torch.Generator().manual_seed(seed)
    wir = []
    prev = NBITS
    for _ in range(DEPTH):
        layer = [(int(torch.randint(prev, (1,), generator=g)),
                  int(torch.randint(prev, (1,), generator=g))) for _ in range(WIDTH)]
        wir.append(layer); prev = WIDTH
    readout = (int(torch.randint(prev, (1,), generator=g)),
               int(torch.randint(prev, (1,), generator=g)))
    return wir, readout


def forward(params, wiring, readout, harden=False):
    cur = [X[:, k] for k in range(NBITS)]
    picks = []
    for li, layer_logits in enumerate(params):
        nxt = []; lpicks = []
        for h in range(WIDTH):
            ia, ib = wiring[li][h]
            R = relax_all(cur[ia], cur[ib])
            if harden:
                g = int(layer_logits[h].argmax()); lpicks.append(g)
                nxt.append(R[:, g])
            else:
                nxt.append(R @ torch.softmax(layer_logits[h], dim=0))
        picks.append(lpicks); cur = nxt
    ia, ib = readout
    R = relax_all(cur[ia], cur[ib])
    if harden:
        g = int(params[-1].new_zeros(1).long()) if False else int(readout_logits.argmax())
        return R[:, g], picks + [[g]]
    return R @ torch.softmax(readout_logits, dim=0), picks


def run(seed, steps=4000, lr=0.05):
    global readout_logits
    torch.manual_seed(seed)
    wiring, readout = build_wiring(seed)
    params = [torch.randn(WIDTH, 16, requires_grad=True) for _ in range(DEPTH)]
    readout_logits = torch.randn(16, requires_grad=True)
    opt = torch.optim.Adam(params + [readout_logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        y, _ = forward(params, wiring, readout, harden=False)
        loss = ((y - T) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        yh, picks = forward(params, wiring, readout, harden=True)
        acc = ((yh > 0.5).double() == T).double().mean().item()
        real = sum(0 if g in {0,3,5,15,12,10} else 1 for lp in picks for g in lp)
    return acc, real, picks


print(f"task: ((x0^x1)|(x2&x3)), 4-bit, {DEPTH} hidden layers x {WIDTH} neurons")
best = (0.0, 999, None)
for s in range(12):
    acc, real, picks = run(s)
    if (acc, -real) > (best[0], -best[1]):
        best = (acc, real, picks)
    print(f"  seed {s}: acc {acc:.2f}  real_gates {real}")
print(f"\nBEST: acc {best[0]:.2f}  real_gates {best[1]}")
print("hardened picks per layer:", best[2])
print("=> DEEP DLGN learns + hardens" if best[0] == 1.0 else "=> FAILED to reach exact")
