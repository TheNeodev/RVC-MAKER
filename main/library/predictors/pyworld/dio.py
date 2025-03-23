import math
import torch

import numpy as np


def dio(x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, target_fs=4000, frame_period=10, allowed_range=0.1, device="cuda"):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    temporal_positions = torch.arange(0, int(1000 * x.numel() / fs / frame_period + 1), device=device, dtype=torch.float32) * (frame_period / 1000)

    y = torch.tensor(decimate(x.cpu().numpy(), int(fs / target_fs)), dtype=torch.float32, device=device)
    raw_f0_candidate, raw_stability = get_candidate_and_stability(temporal_positions, (f0_floor * (2.0 ** ((torch.arange(0, math.ceil(math.log2(f0_ceil / f0_floor) * channels_in_octave), device=device, dtype=torch.float32) + 1) / channels_in_octave))), y.numel(), target_fs, get_spectrum(y, target_fs, f0_floor, device), f0_floor, f0_ceil, device)

    return fix_f0_contour(sort_candidates(raw_f0_candidate, raw_stability), frame_period, f0_floor, allowed_range, device).cpu().numpy(), temporal_positions.cpu().numpy()

def interp1d(x, y, x_new):
    indices_clipped = torch.clamp(torch.searchsorted(x, x_new), 1, len(x)-1)

    x0 = x[indices_clipped - 1]
    y0 = y[indices_clipped - 1]

    return y0 + ((y[indices_clipped] - y0) / (x[indices_clipped] - x0 + 1e-12)) * (x_new - x0)

def nuttall(N, device):
    return torch.matmul(torch.tensor([0.355768, -0.487396, 0.144232, -0.012604], device=device, dtype=torch.float32), torch.cos(torch.arange(4, device=device, dtype=torch.float32).unsqueeze(1) * (2 * math.pi * torch.arange(N, device=device, dtype=torch.float32) / (N - 1)).unsqueeze(0)))

def ZeroCrossingEngine(x, fs, device):
    y = torch.empty_like(x)
    y[:-1] = x[1:]
    y[-1] = x[-1]

    negative_going_points = torch.arange(1, x.numel() + 1, device=device) * ((y * x < 0) & (y < x)).to(torch.int64)
    edge_list = negative_going_points[negative_going_points > 0]
    
    if edge_list.numel() < 2: return torch.tensor([], device=device), torch.tensor([], device=device)

    x_edge_prev = x[(edge_list - 1).long()]
    fine_edge_list = edge_list.to(torch.float32) - x_edge_prev / (x[edge_list.long()] - x_edge_prev + 1e-7)
    
    return (fine_edge_list[:-1] + fine_edge_list[1:]) / (2 * fs), fs / (fine_edge_list[1:] - fine_edge_list[:-1] + 1e-7)

def get_spectrum(x, fs, lowest_f0, device):
    fft_size = 2 ** math.ceil(math.log2(x.numel() + int(fs / lowest_f0 / 2 + 0.5) * 4))
    cutoff_in_sample = int(fs / 50 + 0.5)

    low_cut_filter = torch.hann_window(2 * cutoff_in_sample + 3, periodic=True, device=device)[1:-1]
    low_cut_filter = -low_cut_filter / torch.sum(low_cut_filter)
    low_cut_filter[cutoff_in_sample] += 1
    
    low_cut_filter = torch.cat([low_cut_filter, torch.zeros(fft_size - low_cut_filter.numel(), device=device)])
    low_cut_filter = torch.cat([low_cut_filter[cutoff_in_sample:], low_cut_filter[:cutoff_in_sample]])

    return torch.fft.fft(x, n=fft_size) * torch.fft.fft(low_cut_filter, n=fft_size)

def get_f0_candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions, device):
    interp_f0_list = torch.zeros((4, temporal_positions.numel()), device=device)

    if neg_loc.numel() > 2 and pos_loc.numel() > 2 and peak_loc.numel() > 2 and dip_loc.numel() > 2:
        interp_f0_list[0, :] = interp1d(neg_loc, neg_f0, temporal_positions)
        interp_f0_list[1, :] = interp1d(pos_loc, pos_f0, temporal_positions)
        
        interp_f0_list[2, :] = interp1d(peak_loc, peak_f0, temporal_positions)
        interp_f0_list[3, :] = interp1d(dip_loc, dip_f0, temporal_positions)

        interpolated_f0 = torch.mean(interp_f0_list, dim=0)
        f0_deviations = torch.std(interp_f0_list, dim=0, unbiased=True)
    else:
        interpolated_f0 = torch.zeros_like(temporal_positions)
        f0_deviations = torch.ones_like(temporal_positions) * 1000
        
    return interpolated_f0, f0_deviations

def get_raw_event(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil, device):
    low_pass_filter = nuttall(int(fs / boundary_f0 / 2 + 0.5) * 4, device)
    filtered_signal = torch.fft.ifft(torch.fft.fft(low_pass_filter, n=y_spectrum.numel()) * y_spectrum).real[(torch.arange(1, y_length+1, device=device) + torch.argmax(low_pass_filter)).long()]

    neg_loc, neg_f0 = ZeroCrossingEngine(filtered_signal, fs, device)
    pos_loc, pos_f0 = ZeroCrossingEngine(-filtered_signal, fs, device)

    diff_signal = torch.diff(filtered_signal)

    peak_loc, peak_f0 = ZeroCrossingEngine(diff_signal, fs, device)
    dip_loc, dip_f0 = ZeroCrossingEngine(-diff_signal, fs, device)

    f0_candidate, f0_deviations = get_f0_candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions, device)
    f0_candidate = torch.where(f0_candidate > boundary_f0, torch.tensor(0.0, device=device), f0_candidate)
    f0_candidate = torch.where(f0_candidate < (boundary_f0 / 2), torch.tensor(0.0, device=device), f0_candidate)
    f0_candidate = torch.where(f0_candidate > f0_ceil, torch.tensor(0.0, device=device), f0_candidate)
    f0_candidate = torch.where(f0_candidate < f0_floor, torch.tensor(0.0, device=device), f0_candidate)

    return f0_candidate, torch.where(f0_candidate == 0, torch.tensor(100000.0, device=device), f0_deviations)

def get_candidate_and_stability(temporal_positions, boundary_f0_list, y_length, fs, y_spectrum, f0_floor, f0_ceil, device):
    num_frames = temporal_positions.numel()
    num_boundaries = boundary_f0_list.numel()

    raw_f0_candidate = torch.zeros((num_boundaries, num_frames), dtype=torch.float32, device=device)
    raw_f0_stability = torch.zeros((num_boundaries, num_frames), dtype=torch.float32, device=device)

    for i in range(num_boundaries):
        interpolated_f0, f0_deviations = get_raw_event(boundary_f0_list[i], fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil, device)
        raw_f0_stability[i, :] = torch.exp(- (f0_deviations / torch.maximum(interpolated_f0, torch.tensor(1e-7, device=device))))
        raw_f0_candidate[i, :] = interpolated_f0

    return raw_f0_candidate, raw_f0_stability

def sort_candidates(f0_candidate_map, stability_map):
    sorted_indices = torch.argsort(stability_map, dim=0, descending=True)
    _, num_frames = f0_candidate_map.shape
    f0_candidates = torch.zeros_like(f0_candidate_map)
    
    for i in range(num_frames):
        f0_candidates[:, i] = f0_candidate_map[sorted_indices[:, i], i]
        
    return f0_candidates

def select_best_f0(current_f0, past_f0, candidates, allowed_range):
    reference_f0 = (current_f0 * 3 - past_f0) / 2
    best_f0 = candidates[torch.argmin(torch.abs(reference_f0 - candidates))]
    
    if torch.abs(1 - best_f0 / (reference_f0 + 1e-12)) > allowed_range: best_f0 = 0
        
    return best_f0

def fix_step1(f0_candidates, voice_range_minimum, allowed_range):
    f0_base = f0_candidates[0, :].clone()
    f0_base[:voice_range_minimum] = 0
    f0_base[-voice_range_minimum:] = 0

    f0_step1 = f0_base.clone()
    rounding_f0_base = torch.round(f0_base * 1e6) / 1e6
    
    for i in range(voice_range_minimum, f0_base.numel()):
        if torch.abs((rounding_f0_base[i] - rounding_f0_base[i - 1]) / (rounding_f0_base[i] + 1e-6)) > allowed_range: f0_step1[i] = 0
            
    return f0_step1

def fix_step2(f0_step1, voice_range_minimum):
    f0_step2 = f0_step1.clone()
    half_range = int((voice_range_minimum - 1) // 2)
    
    for i in range(half_range, f0_step1.numel() - half_range):
        for j in range(-half_range, half_range+1):
            if f0_step1[i + j] == 0:
                f0_step2[i] = 0
                break
            
    return f0_step2

def fix_step3(f0_step2, f0_candidates, section_list, allowed_range, device):
    f0_step3 = f0_step2.clone()
    num_sections = section_list.shape[0]
    
    for i in range(num_sections):
        for j in range(int(section_list[i, 1]), (f0_step3.numel() - 1) if i == num_sections - 1 else int(section_list[i+1, 0]) + 1):
            if j + 1 < f0_step3.numel():
                f0_step3[j + 1] = select_best_f0(f0_step3[j], f0_step3[j - 1] if j - 1 >= 0 else f0_step3[j], f0_candidates[:, j + 1], allowed_range)
                if f0_step3[j + 1] == 0: break
                
    return f0_step3

def fix_step4(f0_step3, f0_candidates, section_list, allowed_range, device):
    f0_step4 = f0_step3.clone()

    for i in range(section_list.shape[0] - 1, -1, -1):
        for j in range(int(section_list[i, 0]), (1 if i == 0 else int(section_list[i-1, 1])) - 1, -1):
            f0_step4[j - 1] = select_best_f0(f0_step4[j], f0_step4[j + 1] if j + 1 < f0_step4.numel() else f0_step4[j], f0_candidates[:, j - 1], allowed_range)
            if f0_step4[j - 1] == 0: break

    return f0_step4

def count_voiced_sections(f0, device):
    vuv = f0.clone()
    vuv[vuv != 0] = 1

    diff_vuv = torch.diff(vuv)
    boundaries = torch.cat((torch.tensor([0], device=device), (diff_vuv != 0).nonzero(as_tuple=False).squeeze(), torch.tensor([vuv.numel()-2], device=device)))
    
    if boundaries.numel() < 2: return torch.empty((0,2), device=device)
    
    first_section = torch.ceil(-0.5 * diff_vuv[boundaries[1].long()])
    num_sections = int(torch.floor((boundaries.numel() - (1 - first_section)) / 2).item())
    voiced_section_list = torch.zeros((num_sections, 2), device=device)

    for i in range(num_sections):
        voiced_section_list[i, :] = torch.tensor([1 + boundaries[int((i - 1) * 2 + 1 + (1 - first_section)) + 1], boundaries[int((i * 2) + (1 - first_section)) + 1]], device=device)

    return voiced_section_list

def fix_f0_contour(f0_candidates, frame_period, f0_floor, allowed_range, device):
    voice_range_minimum = int(1 / (frame_period / 1000) / f0_floor + 0.5) * 2 + 1

    f0_step2 = fix_step2(fix_step1(f0_candidates, voice_range_minimum, allowed_range), voice_range_minimum)
    section_list = count_voiced_sections(f0_step2, device)

    return fix_step4(fix_step3(f0_step2, f0_candidates, section_list, allowed_range, device), f0_candidates, section_list, allowed_range, device).clone()

def FilterForDecimate(x, r):
    a, b = np.zeros(3), np.zeros(2)

    if r == 11:
        a[0] = 2.450743295230728
        a[1] = -2.06794904601978
        a[2] = 0.59574774438332101
        b[0] = 0.0026822508007163792
        b[1] = 0.0080467524021491377
    elif r == 12:
        a[0] = 2.4981398605924205
        a[1] = -2.1368928194784025
        a[2] = 0.62187513816221485
        b[0] = 0.0021097275904709001
        b[1] = 0.0063291827714127002
    elif r == 10:
        a[0] = 2.3936475118069387
        a[1] = -1.9873904075111861
        a[2] = 0.5658879979027055
        b[0] = 0.0034818622251927556
        b[1] = 0.010445586675578267
    elif r == 9:
        a[0] = 2.3236003491759578
        a[1] = -1.8921545617463598
        a[2] = 0.53148928133729068
        b[0] = 0.0046331164041389372
        b[1] = 0.013899349212416812
    elif r == 8:
        a[0] = 2.2357462340187593
        a[1] = -1.7780899984041358
        a[2] = 0.49152555365968692
        b[0] = 0.0063522763407111993
        b[1] = 0.019056829022133598
    elif r == 7:
        a[0] = 2.1225239019534703
        a[1] = -1.6395144861046302
        a[2] = 0.44469707800587366
        b[0] = 0.0090366882681608418
        b[1] = 0.027110064804482525
    elif r == 6:
        a[0] = 1.9715352749512141
        a[1] = -1.4686795689225347
        a[2] = 0.3893908434965701
        b[0] = 0.013469181309343825
        b[1] = 0.040407543928031475
    elif r == 5:
        a[0] = 1.7610939654280557
        a[1] = -1.2554914843859768
        a[2] = 0.3237186507788215
        b[0] = 0.021334858522387423
        b[1] = 0.06400457556716227
    elif r == 4:
        a[0] = 1.4499664446880227
        a[1] = -0.98943497080950582
        a[2] = 0.24578252340690215
        b[0] = 0.036710750339322612
        b[1] = 0.11013225101796784
    elif r == 3:
        a[0] = 0.95039378983237421
        a[1] = -0.67429146741526791
        a[2] = 0.15412211621346475
        b[0] = 0.071221945171178636
        b[1] = 0.21366583551353591
    elif r == 2:
        a[0] = 0.041156734567757189
        a[1] = -0.42599112459189636
        a[2] = 0.041037215479961225
        b[0] = 0.16797464681802227
        b[1] = 0.50392394045406674
    else: a[0] = a[1] = a[2] = b[0] = b[1] = 0.0

    w = np.zeros(3)
    y_prime = np.zeros_like(x)

    for i in range(len(x)):
        wt = x[i] + a[0] * w[0] + a[1] * w[1] + a[2] * w[2]
        y_prime[i] = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2]
        w[2] = w[1]
        w[1] = w[0]
        w[0] = wt

    return y_prime

def decimate(x, r):
    y = []
    kNFact = 9
    x_length = len(x)

    tmp1 = np.zeros(x_length + kNFact * 2)
    tmp2 = np.zeros(x_length + kNFact * 2)

    for i in range(kNFact):
        tmp1[i] = 2 * x[0] - x[kNFact - i]

    for i in range(kNFact, kNFact + x_length):
        tmp1[i] = x[i - kNFact]

    for i in range(kNFact + x_length, 2 * kNFact + x_length):
        tmp1[i] = 2 * x[-1] - x[x_length - 2 - (i - (kNFact + x_length))]

    tmp2 = FilterForDecimate(tmp1, r)
    for i in range(2 * kNFact + x_length):
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]

    tmp2 = FilterForDecimate(tmp1, r)
    for i in range(2 * kNFact + x_length):
        tmp1[i] = tmp2[2 * kNFact + x_length - i - 1]

    nbeg = int(r - r * np.ceil(x_length / r + 1) + x_length)
    for i in range(nbeg, x_length + kNFact, r):
        y.append(tmp1[i + kNFact - 1])

    return np.array(y)