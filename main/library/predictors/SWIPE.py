import torch

import numpy as np

from math import sqrt
from matplotlib import mlab
from scipy import interpolate
from decimal import Decimal, ROUND_HALF_UP

def swipe(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10, sTHR=0.3, device="cpu"):
    plim = torch.tensor([f0_floor, f0_ceil], dtype=torch.float32, device=device)
    t = torch.arange(0, int(1000 * len(x) / fs / (frame_period) + 1)) * (frame_period / 1000)

    log2pc = torch.arange(torch.log2(plim[0]) * 96, torch.log2(plim[-1]) * 96)
    log2pc *= (1 / 96)

    pc = 2 ** log2pc
    S = torch.zeros((len(pc), len(t))) 

    logWs = [round_matlab(elm.item()) for elm in torch.log2(4 * 2 * fs / plim)]
    ws = 2 ** torch.arange(logWs[0], logWs[1] - 1, -1).cpu().numpy()
    p0 = 4 * 2 * fs / ws 

    d = 1 + log2pc - torch.log2(4 * 2 * fs / torch.tensor(ws[0], dtype=torch.float32, device=device))
    fERBs = erbs2hz(np.arange(hz2erbs(pc[0] / 4), hz2erbs(torch.tensor(fs / 2, dtype=torch.float32, device=device)), 0.1))

    for i in range(len(ws)):
        dn = round_matlab(4 * fs / p0[i].item()) 
        X, f, ti = mlab.specgram(x=np.r_[np.zeros(int(ws[i] / 2)), np.r_[x, np.zeros(int(dn + ws[i] / 2))]], NFFT=ws[i], Fs=fs, window=np.hanning(ws[i] + 2)[1:-1], noverlap=max(0, np.round(ws[i] - dn)), mode='complex')

        ti = np.r_[0, ti[:-1]]
        M = np.maximum(0, interpolate.interp1d(f, np.abs(X.T), kind='cubic')(fERBs)).T

        if i == len(ws) - 1:
            j = torch.where(d - (i + 1) > -1)[0]
            k = torch.where(d[j] - (i + 1) < 0)[0]
        elif i == 0:
            j = torch.where(d - (i + 1) < 1)[0]
            k = torch.where(d[j] - (i + 1) > 0)[0]
        else:
            j = torch.where(torch.abs(d - (i + 1)) < 1)[0]
            k = torch.arange(len(j))

        Si = pitchStrengthAllCandidates(fERBs, torch.tensor(np.sqrt(M), dtype=torch.float32, device=device), pc[j], device=device)
        Si = torch.tensor(interpolate.interp1d(ti, Si, bounds_error=False, fill_value='nan')(t) if Si.shape[1] > 1 else torch.full((len(Si), len(t)), torch.nan), dtype=torch.float32, device=device)

        mu = torch.ones(j.shape)
        mu[k] = 1 - torch.abs(d[j[k]] - i - 1)

        S[j, :] = S[j, :].to(dtype=torch.float32) + torch.tile(mu.reshape(-1, 1).to(dtype=torch.float32), (1, Si.shape[1])) * Si.to(dtype=torch.float32)

    p = torch.full((S.shape[1], 1), torch.nan)
    s = torch.full((S.shape[1], 1), torch.nan)

    for j in range(S.shape[1]):
        s[j] = torch.max(S[:, j])
        i = torch.argmax(S[:, j])

        if s[j] < sTHR: continue

        if i == 0: p[j] = pc[0]
        elif i == len(pc) - 1: p[j] = pc[0]
        else:
            I = torch.arange(i-1, i+2)
            tc = 1 / pc[I]

            ntc = (tc / tc[1] - 1) * 2 * torch.pi
            idx = torch.isfinite(S[I, j])

            c = torch.zeros(len(ntc))
            c += torch.nan
            I_ = I[idx]

            if len(I_) < 2: c[idx] = torch.tensor((S[I, j])[0] / ntc[0], dtype=torch.float32, device=device)
            else: c[idx] = torch.tensor(np.polyfit(ntc[idx], (S[I_, j]), 2), dtype=torch.float32, device=device)

            pval = torch.tensor(np.polyval(c, ((1 / (2 ** torch.arange(torch.log2(pc[I[0]]), torch.log2(pc[I[2]]) + 1 / 12 / 64,1 / 12 / 64))) / tc[1] - 1) * 2 * torch.pi), dtype=torch.float32, device=device)

            s[j] = torch.max(pval)
            p[j] = 2 ** (torch.log2(pc[I[0]]) + (torch.argmax(pval)) / 12 / 64)

    p = p.flatten()
    p[torch.isnan(p)] = 0

    return p.cpu().numpy().astype(np.float32), t.cpu().numpy().astype(np.float32)

def round_matlab(n):
    return int(Decimal(n).quantize(0, ROUND_HALF_UP))

def pitchStrengthAllCandidates(f, L, pc, device="cpu"):
    den = torch.sqrt(torch.sum(L * L, axis=0))
    den = torch.where(den == 0, 2.220446049250313e-16, den)
    L = L / den

    S = torch.zeros((len(pc), L.shape[1]))
    for j in range(len(pc)):
        S[j,:] = pitchStrengthOneCandidate(f, L, pc[j], device=device)

    return S

def pitchStrengthOneCandidate(f, L, pc, device="cpu"):
    k = torch.zeros(len(f)) 
    q = f / pc 

    for i in ([1] + sieve(int(torch.fix(f[-1] / pc - 0.75)))):
        a = torch.abs(q - i)
        p = a < 0.25
        
        k[p] = torch.cos(2 * torch.pi * q[p].float())

        v = torch.logical_and((0.25 < a), (a < 0.75))
        k[v] = k[v] + torch.cos(2 * torch.pi * q[v].float()) / 2

    k *= torch.sqrt(1 / torch.tensor(f, dtype=torch.float32, device=device))
    k /= torch.linalg.norm(k[k>0])

    return k.to(dtype=torch.float32).to(device) @ L.to(dtype=torch.float32).to(device)

def hz2erbs(hz):
    return 21.4 * torch.log10(1 + hz / 229)

def erbs2hz(erbs):
    return (10 ** (erbs / 21.4) - 1) * 229

def sieve(n):
    primes = list(range(2, n + 1))
    num = 2

    while num < sqrt(n):
        i = num
        while i <= n:
            i += num
            if i in primes: primes.remove(i)
                
        for j in primes:
            if j > num:
                num = j
                break

    return primes