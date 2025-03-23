import math
import torch

import torch.nn.functional as F
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
EPS = 1e-18

def harvest(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10, device="cpu"):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    basic_temporal_positions = torch.arange(0, int(1000 * len(x) / fs + 1), device=device, dtype=torch.float32) / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9

    y, actual_fs = CalculateDownsampledSignal(x, fs, 8000, device=device)

    f0_candidates, number_of_candidates = DetectCandidates(CalculateCandidates(basic_temporal_positions.shape[0], torch.tensor([f0_floor_adjusted * (2.0 ** ((i + 1) / channels_in_octave)) for i in range(int(math.ceil(math.log2((f0_ceil * 1.1) / f0_floor_adjusted) * channels_in_octave) + 1))], device=device, dtype=torch.float32), len(y), basic_temporal_positions, actual_fs, torch.fft.fft(y, n=int(2 ** math.ceil(math.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))), f0_floor, f0_ceil, device=device), device=device)
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates, device=device)

    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs, basic_temporal_positions, f0_candidates, f0_floor, f0_ceil, device=device)
    f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    smoothed_f0 = SmoothF0(FixF0Contour(f0_candidates, f0_candidates_score, device=device), device=device)
    temporal_positions = torch.arange(0, int(1000 * len(x) / fs / frame_period + 1), device=device, dtype=torch.float32) * (frame_period / 1000)
    
    return smoothed_f0[torch.clamp(round_matlab(temporal_positions * 1000).to(torch.long), max=len(smoothed_f0) - 1)].cpu().numpy(), temporal_positions.cpu().numpy()

def CalculateDownsampledSignal(x, fs, target_fs, device="cpu"):
    decimation_ratio = int(fs / target_fs + 0.5)

    if fs <= target_fs:
        y = x.clone()
        actual_fs = fs
    else:
        offset = int(math.ceil(140 / decimation_ratio) * decimation_ratio)
        actual_fs = fs / decimation_ratio
        y_decimated = decimate_matlab(torch.cat((torch.ones(offset, device=device) * x[0], x, torch.ones(offset, device=device) * x[-1])), decimation_ratio, device=device)
        start = int(offset / decimation_ratio)
        y = y_decimated[start:-start]

    return y - torch.mean(y), actual_fs

def CalculateCandidates(number_of_frames, boundary_f0_list, y_length, temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil, device="cpu"):
    raw_f0_candidates = torch.zeros((len(boundary_f0_list), number_of_frames), device=device)
    for i in range(len(boundary_f0_list)):
        raw_f0_candidates[i, :] = CalculateRawEvent(boundary_f0_list[i].item(), actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil, device=device)

    return raw_f0_candidates

def DetectCandidates(raw_f0_candidates, device="cpu"):
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    f0_candidates = torch.zeros((int(number_of_channels / 10 + 0.5), number_of_frames), device=device)
    number_of_candidates = 0

    for i in range(number_of_frames):
        tmp = raw_f0_candidates[:, i].clone()
        tmp[tmp > 0] = 1

        tmp[0] = 0
        tmp[-1] = 0

        tmp_diff = tmp[1:] - tmp[:-1]
        st = (tmp_diff == 1).nonzero(as_tuple=True)[0]
        ed = (tmp_diff == -1).nonzero(as_tuple=True)[0]
        count = 0

        for j in range(len(st)):
            dif = ed[j].item() - st[j].item()
            if dif >= 10:
                f0_candidates[count, i] = torch.mean(raw_f0_candidates[st[j] + 1: ed[j] + 1, i])
                count += 1

        number_of_candidates = max(number_of_candidates, count)

    return f0_candidates, number_of_candidates

def OverlapF0Candidates(f0_candidates, max_candidates, device="cpu"):
    n = 3 
    number_of_candidates = n * 2 + 1

    new_shape1 = f0_candidates.shape[1]
    new_f0_candidates = torch.zeros((number_of_candidates * max_candidates, new_shape1), device=device)
    new_f0_candidates[0, :] = f0_candidates[number_of_candidates - 1, :]

    for i in range(number_of_candidates):
        st1 = max(-(i - n) + 1, 1)
        ed1 = min(-(i - n), 0)

        new_f0_candidates[torch.arange(max_candidates, device=device) + i * max_candidates, st1 - 1:new_shape1 + ed1] = f0_candidates[:max_candidates, -ed1:new_shape1 - (st1 - 1)]

    return new_f0_candidates

def RefineCandidates(x, fs, temporal_positions, f0_candidates, f0_floor, f0_ceil, device="cpu"):
    N, f = f0_candidates.shape

    with mp.Pool(mp.cpu_count()) as pool:
        results = torch.tensor(pool.starmap(GetRefinedF0, [(x, fs, temporal_positions[i].item(), f0_candidates[j, i].item(), f0_floor, f0_ceil, device) for j in range(N) for i in range(f)])).reshape(N, f, 2)

    return results[:,:,0], results[:,:,1]

def round_matlab(x):
    y = x.clone()
    y = torch.where(x > 0, x + 0.5, x - 0.5)
    return torch.floor(y)

def GetRefinedF0(x, fs, current_time, current_f0, f0_floor, f0_ceil, device="cpu"):
    if current_f0 == 0: return 0, 0

    half_window_length = math.ceil(3 * fs / current_f0 / 2)
    fft_size = int(2 ** math.ceil(math.log2(half_window_length * 2 + 1) + 1))

    index_raw = round_matlab((current_time + (torch.arange(-half_window_length, half_window_length + 1, device=device, dtype=torch.float32) / fs)) * fs + 0.001)
    common = math.pi * ((index_raw - 1) / fs - current_time) / ((2 * half_window_length + 1) / fs)
    main_window = 0.42 + 0.5 * torch.cos(2 * common) + 0.08 * torch.cos(4 * common)

    diff_window = torch.empty_like(main_window)
    diff_window[0] = - main_window[1] / 2
    diff_window[-1] = main_window[-2] / 2
    diff = main_window[1:] - main_window[:-1]
    diff_window[1:-1] = - (diff[1:] + diff[:-1]) / 2

    index = torch.clamp(index_raw, 1, len(x)).long() - 1

    spectrum = torch.fft.fft(x[index] * main_window, n=fft_size)
    diff_spectrum = torch.fft.fft(x[index] * diff_window, n=fft_size)

    power_spectrum = torch.abs(spectrum) ** 2
    harmonic_index = torch.arange(1, min(math.floor(fs / 2 / current_f0), 6) + 1, device=device, dtype=torch.float32)
    
    idx = torch.clamp(round_matlab(torch.tensor(current_f0 * fft_size / fs, device=device, dtype=torch.float32) * harmonic_index).long(), max=fft_size-1)
    
    inst_freq = ((torch.arange(fft_size, device=device, dtype=torch.float32) / fft_size + (spectrum.real * diff_spectrum.imag - spectrum.imag * diff_spectrum.real) / (power_spectrum * 2 * math.pi + EPS)) * fs)[idx]
    amplitude_list = torch.sqrt(power_spectrum[idx])
    
    refined_f0 = torch.sum(amplitude_list * inst_freq) / (torch.sum(amplitude_list * harmonic_index) + EPS)
    refined_score = 1 / (1e-12 + torch.mean(torch.abs((inst_freq / harmonic_index - current_f0) / current_f0)))
    
    if refined_f0 < f0_floor or refined_f0 > f0_ceil or refined_score < 2.5: return 0, 0
    return refined_f0.item(), refined_score.item()

def RemoveUnreliableCandidates(f0_candidates, f0_candidates_score):
    new_f0_candidates = f0_candidates.clone()
    new_f0_candidates_score = f0_candidates_score.clone()
    _, num_frames = f0_candidates.shape

    for i in range(1, num_frames - 1):
        current = f0_candidates[:, i]
        nonzero_mask = (current != 0)

        if nonzero_mask.sum() == 0: continue
        curr_vals = current[nonzero_mask]  

        unreliable = torch.minimum((torch.abs(curr_vals.unsqueeze(1) - f0_candidates[:, i + 1].unsqueeze(0)) / (curr_vals.unsqueeze(1) + EPS)).min(dim=1)[0], (torch.abs(curr_vals.unsqueeze(1) - f0_candidates[:, i - 1].unsqueeze(0)) / (curr_vals.unsqueeze(1) + EPS)).min(dim=1)[0]) > 0.05
        indices = torch.where(nonzero_mask)[0]

        new_f0_candidates[indices[unreliable], i] = 0
        new_f0_candidates_score[indices[unreliable], i] = 0

    return new_f0_candidates, new_f0_candidates_score

def SelectBestF0(reference_f0, f0_candidates, allowed_range):
    errors = torch.abs(f0_candidates - reference_f0) / (reference_f0 + EPS)
    min_error, idx = torch.min(torch.where(errors <= allowed_range, errors, torch.full_like(errors, float('inf'))), dim=0)

    return f0_candidates[idx], min_error

def CalculateRawEvent(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil, device="cpu"):
    filter_length_half = int(round(fs / boundary_f0 * 2))
    filtered_signal = torch.real(torch.fft.ifft(torch.fft.fft(nuttall(filter_length_half * 2 + 1, device=device) * torch.cos(2 * math.pi * boundary_f0 * torch.arange(-filter_length_half, filter_length_half + 1, device=device, dtype=torch.float32) / fs), n=y_spectrum.shape[0]) * y_spectrum))

    start = filter_length_half + 1
    filtered_signal = filtered_signal[start: start + y_length]

    neg_loc, neg_f0 = ZeroCrossingEngine(filtered_signal, fs, device=device)
    pos_loc, pos_f0 = ZeroCrossingEngine(-filtered_signal, fs, device=device)

    diff_signal = torch.diff(filtered_signal)

    peak_loc, peak_f0 = ZeroCrossingEngine(diff_signal, fs, device=device)
    dip_loc, dip_f0 = ZeroCrossingEngine(-diff_signal, fs, device=device)

    f0_candidates = GetF0Candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions, device=device)
    return torch.where(((f0_candidates > boundary_f0 * 1.1) | (f0_candidates < boundary_f0 * 0.9) | (f0_candidates > f0_ceil) | (f0_candidates < f0_floor)), torch.tensor(0.0, device=device), f0_candidates)

def ZeroCrossingEngine(x, fs, device="cpu"):
    y = torch.empty_like(x)
    y[:-1] = x[1:]
    y[-1] = x[-1]

    condition = (y * x < 0) & (y < x)
    indices = torch.arange(1, len(x)+1, device=device, dtype=torch.float32) * condition.float()

    edge_list = indices[indices > 0]
    if len(edge_list) < 2: return torch.tensor([]).to(device), torch.tensor([]).to(device)

    edge_list_int = torch.clamp(edge_list.long(), max=len(x)-1)
    fine_edge_list = edge_list - x[edge_list_int - 1] / (x[edge_list_int] - x[edge_list_int - 1] + EPS)

    return (fine_edge_list[:-1] + fine_edge_list[1:]) / 2 / fs, fs / (fine_edge_list[1:] - fine_edge_list[:-1] + EPS)

def FixF0Contour(f0_candidates, f0_candidates_score, device="cpu"):
    return FixStep4(FixStep3(FixStep2(FixStep1(SearchF0Base(f0_candidates, f0_candidates_score, device=device), 0.008), 6), f0_candidates, 0.18, f0_candidates_score, device=device), 9)

def SearchF0Base(f0_candidates, f0_candidates_score, device="cpu"):
    f0_base = torch.zeros(f0_candidates.shape[1], device=device)

    for i in range(len(f0_base)):
        f0_base[i] = f0_candidates[torch.argmax(f0_candidates_score[:, i]), i]

    return f0_base

def FixStep1(f0_base, allowed_range):
    f0_step1 = f0_base.clone()

    if len(f0_step1) > 1: f0_step1[0], f0_step1[1] = 0, 0

    for i in range(2, len(f0_base)):
        if f0_base[i] == 0: continue
        reference_f0 = f0_base[i-1] * 2 - f0_base[i-2]
        if abs((f0_base[i] - reference_f0) / (reference_f0 + EPS)) > allowed_range and abs((f0_base[i] - f0_base[i-1]) / (f0_base[i-1] + EPS)) > allowed_range: f0_step1[i] = 0

    return f0_step1

def FixStep2(f0_step1, voice_range_minimum):
    f0_step2 = f0_step1.clone()
    boundary_list = GetBoundaryList(f0_step1)

    for i in range(1, len(boundary_list) // 2 + 1):
        if boundary_list[2 * i - 1] - boundary_list[2 * i - 2] < voice_range_minimum: f0_step2[boundary_list[2 * i - 2] : boundary_list[2 * i - 1] + 1] = 0

    return f0_step2

def FixStep3(f0_step2, f0_candidates, allowed_range, f0_candidates_score, device="cpu"):
    f0_step3 = f0_step2.clone()
    boundary_list = GetBoundaryList(f0_step2)

    multi_channel_f0 = GetMultiChannelF0(f0_step2, boundary_list)
    ranges = torch.zeros((len(boundary_list)//2, 2), device=device)
    count = -1

    for i in range(1, len(boundary_list) // 2 + 1):
        tmp_range = torch.zeros(2, device=device)

        extended_f0, r1 = ExtendF0(multi_channel_f0[i - 1, :], boundary_list[2 * i - 1].item(), min(len(f0_step2) - 2, boundary_list[2 * i - 1].item() + 100), 1, f0_candidates, allowed_range)
        tmp_range[1] = r1

        extended_f0, r0 = ExtendF0(extended_f0, boundary_list[2 * i - 2].item(), max(1, boundary_list[2 * i - 2].item() - 100), -1, f0_candidates, allowed_range)
        tmp_range[0] = r0

        if 2200 / torch.mean(extended_f0[int(tmp_range[0]): int(tmp_range[1]) + 1] + EPS) < (tmp_range[1] - tmp_range[0]):
            count += 1
            multi_channel_f0[count, :] = extended_f0
            ranges[count, :] = tmp_range

    if count > -1: f0_step3 = MergeF0(multi_channel_f0[0 : count + 1, :], ranges[0 : count + 1, :], f0_candidates, f0_candidates_score)
    return f0_step3

def FixStep4(f0_step3, threshold):
    f0_step4 = f0_step3.clone()
    boundary_list = GetBoundaryList(f0_step3)

    for i in range(1, len(boundary_list) // 2):
        distance = boundary_list[2 * i] - boundary_list[2 * i - 1] - 1
        if distance >= threshold: continue

        tmp0 = f0_step3[boundary_list[2 * i - 1]] + 1
        c = ((f0_step3[boundary_list[2 * i]] - 1) - tmp0) / (distance + 1)
        count = 1

        for j in range(boundary_list[2 * i - 1] + 1, boundary_list[2 * i]):
            f0_step4[j] = tmp0 + c * count
            count += 1

    return f0_step4

def ExtendF0(f0, origin, last_point, shift, f0_candidates, allowed_range):
    extended_f0 = f0.clone()
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin

    count = 0

    if shift == 1: last_point += 1
    elif shift == -1: last_point -= 1

    for i in range(origin, last_point, shift):
        best_f0, _ = SelectBestF0(tmp_f0.item(), f0_candidates[:, i + shift], allowed_range)
        extended_f0[i + shift] = best_f0

        if best_f0 != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else: count += 1

        if count == 4: break

    return extended_f0, shifted_origin

def GetMultiChannelF0(f0, boundary_list, device="cpu"):
    num_channels = len(boundary_list) // 2
    multi_channel_f0 = torch.zeros((num_channels, len(f0)), device=device)

    for i in range(1, num_channels + 1):
        start = boundary_list[2 * i - 2]
        end = boundary_list[2 * i - 1] + 1
        multi_channel_f0[i - 1, start:end] = f0[start:end]

    return multi_channel_f0

def MergeF0(multi_channel_f0, ranges, f0_candidates, f0_candidates_score):
    sorted_order = torch.argsort(ranges[:, 0])
    f0 = multi_channel_f0[sorted_order[0]].clone()
    ranges = ranges.long()

    for i in range(1, multi_channel_f0.shape[0]):
        if ranges[sorted_order[i], 0] - ranges[sorted_order[0], 1] > 0:
            f0[ranges[sorted_order[i], 0]: ranges[sorted_order[i], 1] + 1] = multi_channel_f0[sorted_order[i], ranges[sorted_order[i], 0]: ranges[sorted_order[i], 1] + 1]
            ranges[sorted_order[0], 0] = ranges[sorted_order[i], 0]
            ranges[sorted_order[0], 1] = ranges[sorted_order[i], 1]
        else:
            f0, new_ed = MergeF0Sub(f0, ranges[sorted_order[0], 0], ranges[sorted_order[0], 1], multi_channel_f0[sorted_order[i]], ranges[sorted_order[i], 0], ranges[sorted_order[i], 1], f0_candidates, f0_candidates_score)
            ranges[sorted_order[0], 1] = new_ed

    return f0

def MergeF0Sub(f0_1, st1, ed1, f0_2, st2, ed2, f0_candidates, f0_candidates_score):
    merged_f0 = f0_1.clone()
    st1, st2, ed1, ed2 = int(st1), int(st2), int(ed1), int(ed2)

    if st1 <= st2 and ed1 >= ed2:
        new_ed = ed1
        return merged_f0, new_ed
    
    new_ed = ed2
    score1, score2 = 0, 0

    for i in range(st2, ed1 + 1):
        score1 += SerachScore(f0_1[i].item(), f0_candidates[:, i], f0_candidates_score[:, i])
        score2 += SerachScore(f0_2[i].item(), f0_candidates[:, i], f0_candidates_score[:, i])

    if score1 > score2: merged_f0[ed1: ed2 + 1] = f0_2[ed1: ed2 + 1]
    else: merged_f0[st2: ed2 + 1] = f0_2[st2: ed2 + 1]

    return merged_f0, new_ed

def SerachScore(f0, f0_candidates, f0_candidates_score):
    score = 0

    for i in range(f0_candidates.shape[0]):
        if f0 == f0_candidates[i].item() and score < f0_candidates_score[i].item(): score = f0_candidates_score[i].item()

    return score

def GetF0Candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions, device="cpu"):
    interpolated_f0_list = torch.zeros((4, len(temporal_positions)), device=device)

    if max(len(neg_loc)-2, 0) * max(len(pos_loc)-2, 0) * max(len(peak_loc)-2, 0) * max(len(dip_f0)-2, 0) > 0:
        interpolated_f0_list[0, :] = linear_interpolate(neg_loc, neg_f0, temporal_positions)
        interpolated_f0_list[1, :] = linear_interpolate(pos_loc, pos_f0, temporal_positions)
        interpolated_f0_list[2, :] = linear_interpolate(peak_loc, peak_f0, temporal_positions)
        interpolated_f0_list[3, :] = linear_interpolate(dip_loc, dip_f0, temporal_positions)
        interpolated_f0 = torch.mean(interpolated_f0_list, dim=0)
    else: interpolated_f0 = torch.zeros_like(temporal_positions)

    return interpolated_f0

def interp1d(x, y, x_new):
    indices_clipped = torch.clamp(torch.searchsorted(x, x_new), 1, len(x)-1)

    x0 = x[indices_clipped - 1]
    y0 = y[indices_clipped - 1]

    return y0 + ((y[indices_clipped] - y0) / (x[indices_clipped] - x0 + 1e-12)) * (x_new - x0)

def linear_interpolate(x, y, xi):
    sorted_indices = torch.argsort(x)
    return interp1d(x[sorted_indices], y[sorted_indices], xi)

def SmoothF0(f0, device="cpu"):
    zeros300 = torch.zeros(300, device=device)
    smoothed_f0 = torch.cat((zeros300, f0, zeros300))
    boundary_list = GetBoundaryList(smoothed_f0)

    for i in range(1, len(boundary_list) // 2 + 1):
        smoothed_f0[boundary_list[2 * i - 2] : boundary_list[2 * i - 1] + 1] = FilterF0(GetMultiChannelF0(smoothed_f0, boundary_list, device=device)[i - 1], boundary_list[2 * i - 2], boundary_list[2 * i - 1], torch.tensor([0.0078202080334971724, 0.015640416066994345, 0.007822412033497172], device=device), torch.tensor([1.0, -1.7347257688092754, 0.76600660094326412], device=device))[boundary_list[2 * i - 2] : boundary_list[2 * i - 1] + 1]

    return smoothed_f0[300: -300]

def FilterF0(f0_contour, st, ed, b, a):
    smoothed_f0 = f0_contour.clone()
    smoothed_f0[0 : st] = smoothed_f0[st]
    smoothed_f0[ed + 1: ] = smoothed_f0[ed]

    smoothed_f0 = lfilter(b, a, smoothed_f0)
    smoothed_f0 = torch.flip(lfilter(b, a, torch.flip(smoothed_f0, dims=[0])), dims=[0])

    smoothed_f0[0 : st] = 0
    smoothed_f0[ed + 1: ] = 0

    return smoothed_f0

def lfilter(b, a, x):
    y = torch.zeros_like(x)
    nb, na = len(b), len(a)

    for i in range(len(x)):
        acc = 0.0

        for j in range(nb):
            if i - j >= 0: acc += b[j] * x[i - j]

        for j in range(1, na):
            if i - j >= 0: acc -= a[j] * y[i - j]

        y[i] = acc / a[0]

    return y

def nuttall(N, device="cpu"):
    return torch.sum(torch.tensor([0.355768, -0.487396, 0.144232, -0.012604], device=device, dtype=torch.float32).view(-1,1) * torch.cos(2 * math.pi * torch.arange(4, device=device, dtype=torch.float32).view(-1,1) * torch.arange(N, device=device, dtype=torch.float32) / (N-1)), dim=0)

def GetBoundaryList(f0):
    vuv = f0.clone()
    vuv[vuv != 0] = 1
    vuv[0] = 0
    vuv[-1] = 0

    boundary_list = ((vuv[1:] - vuv[:-1]) != 0).nonzero(as_tuple=True)[0]
    boundary_list = boundary_list.clone()
    boundary_list[0::2] = boundary_list[0::2] + 1

    return boundary_list

def decimate_matlab(x, q, device="cpu"):
    kernel = torch.ones(q, device=device, dtype=torch.float32) / q
    pad = q // 2
    y = F.conv1d(F.pad(x.view(1, 1, -1), (pad, pad), mode='reflect'), kernel.view(1, 1, -1)).view(-1)
    return y[::q]