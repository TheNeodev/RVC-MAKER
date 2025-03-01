import copy
import math

import numba as nb
import numpy as np
import multiprocessing as mp

from scipy import signal
from scipy.fftpack import fft
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from decimal import Decimal, ROUND_HALF_UP


mp.set_start_method("spawn", force=True)

EPS = 0.00000000000000022204460492503131


def harvest(x, fs, f0_floor=50, f0_ceil=1100, frame_period=10):
    basic_temporal_positions = np.arange(0, int(1000 * len(x) / fs / 1 + 1)) * 1 / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9

    y, actual_fs = CalculateDownsampledSignal(x, fs, 8000)

    f0_candidates, number_of_candidates = DetectCandidates(CalculateCandidates(len(basic_temporal_positions), np.array([f0_floor_adjusted * pow(2.0, (i + 1) / channels_in_octave) for i in range(int(np.ceil(np.log2((f0_ceil * 1.1) / f0_floor_adjusted) * channels_in_octave) + 1))]), len(y), basic_temporal_positions, actual_fs, np.fft.fft(y, int(2 ** np.ceil(np.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))), f0_floor, f0_ceil))
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates)

    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs, basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)
    f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    smoothed_f0 = SmoothF0(FixF0Contour(f0_candidates, f0_candidates_score))
    temporal_positions = np.arange(0, int(1000 * len(x) / fs / frame_period + 1)) * frame_period / 1000

    return np.array(smoothed_f0[np.array(np.minimum(len(smoothed_f0) - 1, round_matlab(temporal_positions * 1000)), dtype=int)], dtype=np.float32), np.array(temporal_positions, dtype=np.float32)

def CalculateDownsampledSignal(x, fs, target_fs):
    decimation_ratio = int(fs / target_fs + 0.5)

    if fs <= target_fs:
        y = copy.deepcopy(x)
        actual_fs = fs
    else:
        offset = int(np.ceil(140 / decimation_ratio) * decimation_ratio)
        actual_fs = fs / decimation_ratio
        y = decimate_matlab(np.append(np.append(np.ones(offset) * x[0], x), np.ones(offset) * x[-1]), decimation_ratio, n = 3)[int(offset / decimation_ratio) : int(-offset / decimation_ratio)]

    y -= np.mean(y)
    return y, actual_fs

def CalculateCandidates(number_of_frames, boundary_f0_list, y_length, temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil):
    raw_f0_candidates = np.zeros((len(boundary_f0_list), number_of_frames))

    for i in range(len(boundary_f0_list)):
        raw_f0_candidates[i, :] = CalculateRawEvent(boundary_f0_list[i], actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil)

    return raw_f0_candidates

def DetectCandidates(raw_f0_candidates):
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    f0_candidates = np.zeros((int(number_of_channels / 10 + 0.5), number_of_frames))

    number_of_candidates = 0
    threshold = 10

    for i in np.arange(number_of_frames):
        tmp = np.array(raw_f0_candidates[:, i])
        tmp[tmp > 0] = 1
        tmp[0] = 0
        tmp[-1] = 0
        tmp = np.diff(tmp)

        st = np.where(tmp == 1)[0]
        ed = np.where(tmp == -1)[0]

        count = 0

        for j in np.arange(len(st)):
            dif = ed[j] - st[j]

            if dif >= threshold:
                f0_candidates[count, i] = np.mean(raw_f0_candidates[st[j] + 1: ed[j] + 1, i])
                count += 1

        number_of_candidates = max(number_of_candidates, count)

    return f0_candidates, number_of_candidates

def OverlapF0Candidates(f0_candidates, max_candidates):
    n = 3 
    number_of_candidates = n * 2 + 1

    new_f0_candidates = np.zeros((number_of_candidates * max_candidates, f0_candidates.shape[1]))
    new_f0_candidates[0, :] = f0_candidates[number_of_candidates - 1, :]

    for i in np.arange(number_of_candidates):
        st1 = max(-(i - n) + 1, 1)
        ed1 = min(-(i - n), 0)
        new_f0_candidates[np.arange(max_candidates) + i * max_candidates, st1 - 1 : new_f0_candidates.shape[1] + ed1] = f0_candidates[np.arange(max_candidates), -ed1 : new_f0_candidates.shape[1] - (st1 - 1)]
        
    return new_f0_candidates

def RefineCandidates(x, fs, temporal_positions, f0_candidates, f0_floor, f0_ceil):
    N, f = f0_candidates.shape

    with mp.Pool(mp.cpu_count()) as pool:
        results = np.array(pool.starmap(GetRefinedF0, [(x, fs, temporal_positions[i], f0_candidates[j, i], f0_floor, f0_ceil) for j in np.arange(N) for i in np.arange(f)]))

    return np.reshape(results[:, 0], [N, f]), np.reshape(results[:, 1], [N, f])

@nb.jit((nb.float64[:],), nopython=True, cache=True)
def round_matlab(x):
    y = x.copy()
    y[x > 0] += 0.5
    y[x <= 0] -= 0.5

    return y

def GetRefinedF0(x, fs, current_time, current_f0, f0_floor, f0_ceil):
    if current_f0 == 0: return 0, 0

    half_window_length = np.ceil(3 * fs / current_f0 / 2)
    fft_size = int(2 ** np.ceil(np.log2((half_window_length * 2 + 1)) + 1))
    index_raw = round_matlab((current_time + (np.arange(-half_window_length, half_window_length + 1) / fs)) * fs + 0.001)
    common = math.pi * ((index_raw - 1) / fs - current_time) / ((2 * half_window_length + 1) / fs)
    main_window = 0.42 + 0.5 * np.cos(2 * common) + 0.08 * np.cos(4 * common)

    diff_window = np.empty_like(main_window)
    diff_window[0] = - main_window[1] / 2
    diff_window[-1] = main_window[-2] / 2
    diff = np.diff(main_window)
    diff_window[1:-1] = - (diff[1:] + diff[:-1]) / 2

    index = (np.maximum(1, np.minimum(len(x), index_raw)) - 1).astype(int)
    spectrum = fft(x[index] * main_window, fft_size)
    diff_spectrum = fft(x[index] * diff_window, fft_size)

    power_spectrum = np.abs(spectrum) ** 2
    number_of_harmonics = min(np.floor(fs / 2 / current_f0), 6) 
    harmonic_index = np.arange(1, number_of_harmonics + 1)

    index = round_matlab(current_f0 * fft_size / fs * harmonic_index).astype(int)
    instantaneous_frequency_list = ((np.arange(fft_size) / fft_size + (spectrum.real * diff_spectrum.imag - spectrum.imag * diff_spectrum.real) / power_spectrum / 2 / math.pi) * fs)[index]
    amplitude_list = np.sqrt(power_spectrum[index])

    refined_f0 = np.sum(amplitude_list * instantaneous_frequency_list) / np.sum(amplitude_list * harmonic_index)
    refined_score = 1 / (0.000000000001 + np.mean(np.abs(((instantaneous_frequency_list / harmonic_index) - current_f0) / current_f0)))

    if refined_f0 < f0_floor or refined_f0 > f0_ceil or refined_score < 2.5: refined_f0 = refined_score = 0

    return refined_f0, refined_score

def RemoveUnreliableCandidates(f0_candidates, f0_candidates_score):
    new_f0_candidates = np.array(f0_candidates)
    new_f0_candidates_score = np.array(f0_candidates_score)

    for i in np.arange(1, f0_candidates.shape[1] - 1):
        for j in np.arange(0, f0_candidates.shape[0]):
            reference_f0 = f0_candidates[j, i]
            if reference_f0 == 0: continue

            _, min_error1 = SelectBestF0(reference_f0, f0_candidates[:, i + 1], 1)
            _, min_error2 = SelectBestF0(reference_f0, f0_candidates[:, i - 1], 1)

            min_error = min([min_error1, min_error2])
            if min_error > 0.05: new_f0_candidates[j, i] = new_f0_candidates_score[j, i] = 0

    return new_f0_candidates, new_f0_candidates_score

@nb.jit((nb.float64, nb.float64[:], nb.float64), nopython=True, cache=True)  
def SelectBestF0(reference_f0, f0_candidates, allowed_range):
    best_f0 = 0
    best_error = allowed_range

    for i in np.arange(len(f0_candidates)):
        tmp = np.abs(reference_f0 - f0_candidates[i]) / reference_f0
        if tmp > best_error: continue

        best_f0 = f0_candidates[i]
        best_error = tmp

    return best_f0, best_error

def CalculateRawEvent(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil):
    filter_length_half = int(Decimal(fs / boundary_f0 * 2).quantize(0, ROUND_HALF_UP))

    filtered_signal = np.real(np.fft.ifft(np.fft.fft(nuttall(filter_length_half * 2 + 1) * np.cos(2 * math.pi * boundary_f0 * np.arange(-filter_length_half, filter_length_half + 1) / fs), len(y_spectrum)) * y_spectrum))
    filtered_signal = filtered_signal[(filter_length_half + 1) + np.arange(y_length)]

    neg_loc, neg_f0 = ZeroCrossingEngine(filtered_signal, fs)
    pos_loc, pos_f0 = ZeroCrossingEngine(-filtered_signal, fs)

    peak_loc, peak_f0 = ZeroCrossingEngine(np.diff(filtered_signal), fs)
    dip_loc, dip_f0 = ZeroCrossingEngine(-np.diff(filtered_signal), fs)

    f0_candidates = GetF0Candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions)
    f0_candidates[f0_candidates > boundary_f0 * 1.1] = f0_candidates[f0_candidates < boundary_f0 * 0.9] = f0_candidates[f0_candidates > f0_ceil] = f0_candidates[f0_candidates < f0_floor] = 0

    return f0_candidates

@nb.jit((nb.float64[:], nb.float64), nopython=True, cache=True)
def ZeroCrossingEngine(x, fs):
    y = np.empty_like(x)
    y[:-1] = x[1:]
    y[-1] = x[-1]

    negative_going_points = np.arange(1, len(x) + 1) * ((y * x < 0) * (y < x))
    edge_list = negative_going_points[negative_going_points > 0]
    fine_edge_list = (edge_list) - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1])

    return (fine_edge_list[:len(fine_edge_list) - 1] + fine_edge_list[1:]) / 2 / fs, fs / np.diff(fine_edge_list)

def FixF0Contour(f0_candidates, f0_candidates_score):
    return FixStep4(FixStep3(FixStep2(FixStep1(SearchF0Base(f0_candidates, f0_candidates_score), 0.008), 6), f0_candidates, 0.18, f0_candidates_score), 9)

def SearchF0Base(f0_candidates, f0_candidates_score):
    f0_base = np.zeros((f0_candidates.shape[1]))

    for i in range(len(f0_base)):
        f0_base[i] = f0_candidates[np.argmax(f0_candidates_score[:, i]), i]

    return f0_base

@nb.jit((nb.float64[:], nb.float64), nopython=True, cache=True)
def FixStep1(f0_base, allowed_range):
    f0_step1 = np.empty_like(f0_base)
    f0_step1[:] = f0_base
    f0_step1[0] = f0_step1[1] = 0

    for i in np.arange(2, len(f0_base)):
        if f0_base[i] == 0: continue

        reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2]
        if np.abs((f0_base[i] - reference_f0) / (reference_f0 + EPS)) > allowed_range and np.abs((f0_base[i] - f0_base[i - 1]) / (f0_base[i - 1] + EPS)) > allowed_range: f0_step1[i] = 0

    return f0_step1

def FixStep2(f0_step1, voice_range_minimum):
    f0_step2 = np.empty_like(f0_step1)
    f0_step2[:] = f0_step1

    boundary_list = GetBoundaryList(f0_step1)

    for i in np.arange(1, len(boundary_list) // 2 + 1):
        if boundary_list[2 * i - 1] - boundary_list[(2 * i) - 2] < voice_range_minimum: f0_step2[boundary_list[(2 * i) - 2] : boundary_list[2 * i - 1] + 1] = 0

    return f0_step2

def FixStep3(f0_step2, f0_candidates, allowed_range, f0_candidates_score):
    f0_step3 = np.array(f0_step2)
    boundary_list = GetBoundaryList(f0_step2)

    multi_channel_f0 = GetMultiChannelF0(f0_step2, boundary_list)
    range = np.zeros((len(boundary_list) // 2, 2))

    count = -1
    for i in np.arange(1, len(boundary_list) // 2 + 1):
        tmp_range = np.zeros(2)

        extended_f0, tmp_range[1] = ExtendF0(multi_channel_f0[i - 1, :], boundary_list[i * 2 - 1], min(len(f0_step2) - 2, boundary_list[i * 2 - 1] + 100), 1, f0_candidates, allowed_range)
        tmp_f0_sequence, tmp_range[0] = ExtendF0(extended_f0, boundary_list[(i * 2) - 2], max(1, boundary_list[(i * 2) - 2] - 100), -1, f0_candidates, allowed_range)

        if 2200 / np.mean(tmp_f0_sequence[int(tmp_range[0]) : int(tmp_range[1]) + 1]) < tmp_range[1] - tmp_range[0]:
            count += 1
            multi_channel_f0[count, :] = tmp_f0_sequence
            range[count, :] = tmp_range

    if count > -1: f0_step3 = MergeF0(multi_channel_f0[0 : count + 1, :], range[0 : count + 1, :], f0_candidates, f0_candidates_score)
    return f0_step3

def FixStep4(f0_step3, threshold):
    f0_step4 = np.empty_like(f0_step3)
    f0_step4[:] = f0_step3

    boundary_list = GetBoundaryList(f0_step3)

    for i in np.arange(1, len(boundary_list) // 2 ):
        distance = boundary_list[2 * i] - boundary_list[2 * i - 1] - 1
        if distance >= threshold: continue

        tmp0 = f0_step3[boundary_list[2 * i - 1]] + 1
        c = ((f0_step3[boundary_list[2 * i]] - 1) - tmp0) / (distance + 1)
        count = 1

        for j in np.arange(boundary_list[2 * i - 1] + 1, boundary_list[2 * i]):
            f0_step4[j] = tmp0 + c * count
            count += 1

    return f0_step4

def ExtendF0(f0, origin, last_point, shift, f0_candidates, allowed_range):
    extended_f0 = np.array(f0)
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin

    count = 0

    if shift == 1: last_point += 1
    elif shift == -1: last_point -= 1

    for i in np.arange(origin, last_point, shift):
        extended_f0[i + shift], _ = SelectBestF0(tmp_f0, f0_candidates[:, i + shift], allowed_range)

        if extended_f0[i + shift] != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else: count += + 1

        if count == 4: break

    return extended_f0, shifted_origin

def GetMultiChannelF0(f0, boundary_list):
    multi_channel_f0 = np.zeros((len(boundary_list) // 2, len(f0)))

    for i in np.arange(1, len(boundary_list) // 2 + 1):
        multi_channel_f0[i - 1, boundary_list[(i * 2) - 2] : boundary_list[i * 2 - 1] + 1] = f0[boundary_list[(i * 2) - 2] : boundary_list[(i * 2) - 1] + 1]

    return multi_channel_f0

def MergeF0(multi_channel_f0, range_, f0_candidates, f0_candidates_score):
    sorted_order = np.argsort(range_[:, 0], axis=0, kind='quicksort')
    f0 = multi_channel_f0[sorted_order[0], :]
    range_ = range_.astype(int)

    for i in np.arange(1, multi_channel_f0.shape[0]):
        if range_[sorted_order[i], 0] - range_[sorted_order[0], 1] > 0:
            f0[range_[sorted_order[i], 0] : range_[sorted_order[i], 1] + 1] = multi_channel_f0[sorted_order[i], range_[sorted_order[i], 0] : range_[sorted_order[i], 1] + 1]
            range_[sorted_order[0], 0] = range_[sorted_order[i], 0]
            range_[sorted_order[0], 1] = range_[sorted_order[i], 1]
        else: f0, range_[sorted_order[0], 1] = MergeF0Sub(f0, range_[sorted_order[0], 0], range_[sorted_order[0], 1], multi_channel_f0[sorted_order[i], :], range_[sorted_order[i], 0], range_[sorted_order[i], 1], f0_candidates, f0_candidates_score)
            
    return f0

def MergeF0Sub(f0_1, st1, ed1, f0_2, st2, ed2, f0_candidates, f0_candidates_score):
    merged_f0 = copy.deepcopy(f0_1)
    st1, st2, ed1, ed2 = int(st1), int(st2), int(ed1), int(ed2)

    if st1 <= st2 and ed1 >= ed2:
        new_ed = ed1
        return merged_f0, new_ed
    
    new_ed = ed2
    score1, score2 = 0, 0

    for i in np.arange(st2, ed1 + 1):
        score1 = score1 + SerachScore(f0_1[i], f0_candidates[:, i], f0_candidates_score[:, i])
        score2 = score2 + SerachScore(f0_2[i], f0_candidates[:, i], f0_candidates_score[:, i])

    if score1 > score2: merged_f0[ed1 : ed2 + 1] = f0_2[ed1 : ed2 + 1]
    else: merged_f0[st2 : ed2 + 1] = f0_2[st2 : ed2 + 1]

    return merged_f0, new_ed

def SerachScore(f0, f0_candidates, f0_candidates_score):
    score = 0

    for i in range(f0_candidates.shape[0]):
        if f0 == f0_candidates[i] and score < f0_candidates_score[i]: score = f0_candidates_score[i]

    return score

def GetF0Candidates(neg_loc, neg_f0, pos_loc, pos_f0, peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions):
    interpolated_f0_list = np.zeros((4, np.size(temporal_positions)))

    if max(0, np.size(neg_loc) - 2) * max(0, np.size(pos_loc) - 2) * max(0, np.size(peak_loc) - 2) * max(0, np.size(dip_f0) - 2) > 0:
        interpolated_f0_list[0, :] = interp1d(neg_loc, neg_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1, :] = interp1d(pos_loc, pos_f0, fill_value='extrapolate')(temporal_positions)

        interpolated_f0_list[2, :] = interp1d(peak_loc, peak_f0, fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3, :] = interp1d(dip_loc, dip_f0, fill_value='extrapolate')(temporal_positions)

        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
    else: interpolated_f0 = temporal_positions * 0

    return interpolated_f0

def SmoothF0(f0):
    smoothed_f0 = np.append(np.append(np.zeros(300), f0), np.zeros(300))
    boundary_list = GetBoundaryList(smoothed_f0)

    for i in np.arange(1, len(boundary_list) // 2 + 1):
        tmp_f0_contour = FilterF0(GetMultiChannelF0(smoothed_f0, boundary_list)[i - 1, :], boundary_list[i * 2 - 2], boundary_list[i * 2 - 1], np.array([0.0078202080334971724, 0.015640416066994345, 0.007822412033497172]), np.array([1.0, -1.7347257688092754, 0.76600660094326412]))
        smoothed_f0[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1] = tmp_f0_contour[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1]

    return smoothed_f0[300 : len(smoothed_f0) - 300]

def FilterF0(f0_contour, st, ed, b, a):
    smoothed_f0 = copy.deepcopy(f0_contour)
    smoothed_f0[0 : st] = smoothed_f0[st]
    smoothed_f0[ed + 1: ] = smoothed_f0[ed]
    smoothed_f0 = lfilter(b, a, lfilter(b, a, smoothed_f0, axis=0)[-1 : : -1], axis=0)[-1 : : -1]
    smoothed_f0[0 : st] = smoothed_f0[ed + 1: ] = 0

    return smoothed_f0

def nuttall(N):
    return np.squeeze(np.asarray(np.array([0.355768, -0.487396, 0.144232, -0.012604]) @ np.cos(np.matrix([0,1,2,3]).T @ np.asmatrix(np.arange(N) * 2 * math.pi / (N-1)))))

def GetBoundaryList(f0):
    vuv = np.array(f0)
    vuv[vuv != 0] = 1
    vuv[0] = vuv[-1] = 0

    boundary_list = np.where(np.diff(vuv) != 0)[0]
    boundary_list[0:: 2] += 1

    return boundary_list

def decimate_matlab(x, q, n=None, axis=-1):
    if not isinstance(q, int): raise TypeError
    if n is not None and not isinstance(n, int): raise TypeError

    system = signal.dlti(*signal.cheby1(n, 0.05, 0.8 / q))
    y = signal.filtfilt(system.num, system.den, x, axis=axis, padlen=3 * (max(len(system.den), len(system.num)) - 1))
    nd = len(y)

    return y[int(q - (q * np.ceil(nd / q) - nd)) - 1::q]