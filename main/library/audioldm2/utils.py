import torch
import librosa
import torchaudio

import numpy as np

def compute_mel_spectrogram(audio, stft_processor):
    return stft_processor.compute_mel_spectrogram(torch.autograd.Variable(torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1), requires_grad=False)).squeeze(0).numpy().astype(np.float32)

def pad_spectrogram(spectrogram, target_length=1024):
    pad_amount = target_length - spectrogram.shape[0]
    spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, pad_amount)) if pad_amount > 0 else spectrogram[:target_length, :]
    
    if spectrogram.size(-1) % 2 != 0: spectrogram = spectrogram[..., :-1]
    return spectrogram

def pad_waveform(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100
    
    if segment_length is None or waveform_length == segment_length: return waveform
    elif waveform_length > segment_length: return waveform[:, :segment_length]
    
    padded_waveform = np.zeros((1, segment_length))
    padded_waveform[:, :waveform_length] = waveform
    return padded_waveform

def normalize(waveform):
    waveform -= np.mean(waveform)
    return (waveform / (np.max(np.abs(waveform)) + 1e-8)) * 0.5

def process_audio(y, sr, segment_length):
    normalized_waveform = normalize(torchaudio.functional.resample(torch.from_numpy(y), orig_freq=sr, new_freq=16000).numpy())[None, ...]
    return 0.5 * (pad_waveform(normalized_waveform, segment_length) / np.max(np.abs(normalized_waveform)))

def load_audio(audio_path, stft_processor, device=None):
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    return pad_spectrogram(torch.FloatTensor(compute_mel_spectrogram(torch.FloatTensor(process_audio(y, sr, int(duration * 102.4) * 160)[0, ...]), stft_processor).T), int(duration * 102.4)).unsqueeze(0).unsqueeze(0).to(device), duration