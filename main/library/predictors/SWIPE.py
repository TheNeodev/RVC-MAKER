import math
import torch
import logging

import numpy as np
import torch.nn.functional as F

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def swipe(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10, sTHR=0.3, device='cuda'):
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    t = torch.arange(0, int(1000 * len(x) / fs / frame_period + 1), device=device, dtype=torch.float32) * (frame_period / 1000)

    plim = torch.tensor([f0_floor, f0_ceil], dtype=torch.float32, device=device)
    log2pc = torch.arange(torch.log2(plim[0]) * 96, torch.log2(plim[1]) * 96, device=device) / 96.0

    pc = torch.pow(2, log2pc)
    S = torch.zeros((len(pc), len(t)), device=device, dtype=torch.float32)
    
    logWs = [int(round(math.log2(4 * 2 * fs / float(val)))) for val in plim.cpu().numpy()]
    ws_vals = 2 ** np.arange(logWs[0], logWs[1] - 1, -1)
    p0 = 4 * 2 * fs / torch.tensor(ws_vals, dtype=torch.float32, device=device)
    
    d = 1 + log2pc - math.log2(4 * 2 * fs / ws_vals[0])
    fERBs = erbs2hz_torch(torch.arange(hz2erbs_torch(pc[0]/4), hz2erbs_torch(torch.tensor(fs/2, device=device)), 0.1, device=device))
    
    for i, ws in enumerate(ws_vals):
        ws = int(ws)
        dn = int(round(4 * fs / p0[i].item()))

        X_abs = torch.abs(torch.stft(torch.cat([torch.zeros(ws // 2, device=device), x_torch, torch.zeros(dn + ws // 2, device=device)]), n_fft=ws, hop_length=dn, window=torch.hann_window(ws, device=device), return_complex=True))
        f = torch.linspace(0, fs/2, steps=X_abs.shape[0], device=device, dtype=torch.float32)
        
        if i == len(ws_vals) - 1:
            j = (d - (i + 1) > -1).nonzero(as_tuple=True)[0]
            k = (d[j] - (i + 1) < 0).nonzero(as_tuple=True)[0]
        elif i == 0:
            j = (d - (i + 1) < 1).nonzero(as_tuple=True)[0]
            k = (d[j] - (i + 1) > 0).nonzero(as_tuple=True)[0]
        else:
            j = (torch.abs(d - (i + 1)) < 1).nonzero(as_tuple=True)[0]
            k = torch.arange(len(j), device=device)
        
        mu = torch.ones(j.shape, device=device, dtype=torch.float32)
        if k.numel() > 0: mu[k] = 1 - torch.abs(d[j][k] - (i+1))
        
        S[j, :] = S[j, :] + mu.unsqueeze(1) * F.interpolate(pitchStrengthAllCandidates(fERBs, torch.sqrt(torch.clamp(torch.stack([linear_interpolate_1d(f, X_abs[:, col], fERBs) for col in range(X_abs.shape[-1])], dim=1) , min=0)), pc[j]).unsqueeze(1), size=len(t), mode='linear', align_corners=True).squeeze(1)

    p = torch.full((S.shape[1],), float('nan'), device=device, dtype=torch.float32)
    s = torch.full((S.shape[1],), float('nan'), device=device, dtype=torch.float32)
    
    for j in range(S.shape[1]):
        s_val, i = torch.max(S[:, j], dim=0)
        s[j] = s_val

        if s_val < sTHR: continue

        if i == 0 or i == len(pc)-1: p[j] = pc[0]
        else:
            I = torch.arange(i-1, i+2, device=device)

            tc = 1 / pc[I]
            ntc = (tc / tc[1] - 1) * 2 * math.pi

            idx = torch.isfinite(S[I, j])
            pval = S[I, j][0] * torch.ones(10, device=device) if I[idx].numel() < 2 else polyval(polyfit(ntc[idx], S[I, j][idx], deg=2), torch.linspace(ntc[0], ntc[-1], steps=10, device=device))

            s[j] = torch.max(pval)
            p[j] = 2 ** (torch.log2(pc[I[0]]) + (torch.argmax(pval).item()) / (12 * 64))
    
    p[torch.isnan(p)] = 0

    return p.cpu().numpy(), t.cpu().numpy()

def hz2erbs_torch(hz):
    return 21.4 * torch.log10(1 + hz / 229)

def erbs2hz_torch(erbs):
    return (torch.pow(10, erbs / 21.4) - 1) * 229

def linear_interpolate_1d(x, y, x_new):
    inds = torch.clamp(torch.searchsorted(x, x_new), 1, len(x)-1)

    x0 = x[inds - 1]
    y0 = y[inds - 1]

    return y0 + ((x_new - x0) / (x[inds] - x0)) * (y[inds] - y0)

def polyfit(x, y, deg=2):
    return torch.linalg.lstsq(torch.stack([x**i for i in range(deg, -1, -1)], dim=1), y).solution

def polyval(coeffs, x):
    deg = coeffs.numel() - 1
    y = torch.zeros_like(x)

    for i, c in enumerate(coeffs):
        y = y + c * x ** (deg - i)

    return y

def sieve(n):
    primes = list(range(2, n+1))
    num = 2

    while num < math.sqrt(n):
        i = num

        while i <= n:
            i += num

            if i in primes: primes.remove(i)
                
        for j in primes:
            if j > num:
                num = j
                break

    return primes

def pitchStrengthOneCandidate(f, L, pc):
    q = f / pc
    k = torch.zeros_like(f)

    for m in [1] + sieve(int(math.floor(f[-1].item() / pc - 0.75))):
        a = torch.abs(q - m)

        mask1 = a < 0.25
        mask2 = (a > 0.25) & (a < 0.75)

        k[mask1] = torch.cos(2 * math.pi * q[mask1])
        k[mask2] = k[mask2] + torch.cos(2 * math.pi * q[mask2]) / 2

    k = k * torch.sqrt(1 / f)
    pos = k > 0

    return torch.matmul(k / torch.norm(k[pos]) if torch.any(pos) else torch.tensor(1.0, device=k.device), L)

def pitchStrengthAllCandidates(f, L, pc_candidates):
    return torch.stack([pitchStrengthOneCandidate(f, L, pc) for pc in pc_candidates], dim=0)