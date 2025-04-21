import re
import os
import sys
import time
import faiss
import torch
import librosa
import logging
import argparse
import warnings
import onnxruntime
import logging.handlers

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from tqdm import tqdm
from scipy import signal
from distutils.util import strtobool

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.algorithm.synthesizers import Synthesizer
from main.library.utils import check_predictors, check_embedders, load_audio, load_embedders_model, cut, restore

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
config = Config()
translations = config.translations
logger = logging.getLogger(__name__)
logger.propagate = False

for l in ["torch", "faiss", "httpx", "fairseq", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers"]:
    logging.getLogger(l).setLevel(logging.ERROR)

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join("assets", "logs", "convert.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--volume_envelope", type=float, default=1)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--embedder_model", type=str, default="contentvec_base")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/output.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--pth_path",  type=str,  required=True)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--split_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_file", type=str, default="")
    parser.add_argument("--f0_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--formant_shifting", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--formant_qfrency", type=float, default=0.8)
    parser.add_argument("--formant_timbre", type=float, default=0.8)

    return parser.parse_args()

def main():
    args = parse_arguments()
    pitch, filter_radius, index_rate, volume_envelope, protect, hop_length, f0_method, input_path, output_path, pth_path, index_path, f0_autotune, f0_autotune_strength, clean_audio, clean_strength, export_format, embedder_model, resample_sr, split_audio, checkpointing, f0_file, f0_onnx, embedders_mode, formant_shifting, formant_qfrency, formant_timbre = args.pitch, args.filter_radius, args.index_rate, args.volume_envelope,args.protect, args.hop_length, args.f0_method, args.input_path, args.output_path, args.pth_path, args.index_path, args.f0_autotune, args.f0_autotune_strength, args.clean_audio, args.clean_strength, args.export_format, args.embedder_model, args.resample_sr, args.split_audio, args.checkpointing, args.f0_file, args.f0_onnx, args.embedders_mode, args.formant_shifting, args.formant_qfrency, args.formant_timbre

    log_data = {translations['pitch']: pitch, translations['filter_radius']: filter_radius, translations['index_strength']: index_rate, translations['volume_envelope']: volume_envelope, translations['protect']: protect, "Hop length": hop_length, translations['f0_method']: f0_method, translations['audio_path']: input_path, translations['output_path']: output_path.replace('wav', export_format), translations['model_path']: pth_path, translations['indexpath']: index_path, translations['autotune']: f0_autotune, translations['clear_audio']: clean_audio, translations['export_format']: export_format, translations['hubert_model']: embedder_model, translations['split_audio']: split_audio, translations['memory_efficient_training']: checkpointing, translations["f0_onnx_mode"]: f0_onnx, translations["embed_mode"]: embedders_mode}

    if clean_audio: log_data[translations['clean_strength']] = clean_strength
    if resample_sr != 0: log_data[translations['sample_rate']] = resample_sr

    if f0_autotune: log_data[translations['autotune_rate_info']] = f0_autotune_strength
    if os.path.isfile(f0_file): log_data[translations['f0_file']] = f0_file

    if formant_shifting:
        log_data[translations['formant_qfrency']] = formant_qfrency
        log_data[translations['formant_timbre']] = formant_timbre

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")
    
    run_convert_script(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, input_path=input_path, output_path=output_path, pth_path=pth_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, split_audio=split_audio, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)

def run_convert_script(pitch=0, filter_radius=3, index_rate=0.5, volume_envelope=1, protect=0.5, hop_length=64, f0_method="rmvpe", input_path=None, output_path="./output.wav", pth_path=None, index_path=None, f0_autotune=False, f0_autotune_strength=1, clean_audio=False, clean_strength=0.7, export_format="wav", embedder_model="contentvec_base", resample_sr=0, split_audio=False, checkpointing=False, f0_file=None, f0_onnx=False, embedders_mode="fairseq", formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    check_predictors(f0_method, f0_onnx); check_embedders(embedder_model, embedders_mode)

    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")):
        logger.warning(translations["provide_file"].format(filename=translations["model"]))
        sys.exit(1)

    cvt = VoiceConverter(pth_path, 0)
    start_time = time.time()

    pid_path = os.path.join("assets", "convert_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    if os.path.isdir(input_path):
        logger.info(translations["convert_batch"])
        audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

        if not audio_files: 
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(translations["found_audio"].format(audio_files=len(audio_files)))

        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

            logger.info(f"{translations['convert_audio']} '{audio_path}'...")
            if os.path.exists(output_audio): os.remove(output_audio)
            cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=audio_path, audio_output_path=output_audio, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre, split_audio=split_audio)

        logger.info(translations["convert_batch_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}", output_path=output_path.replace('wav', export_format)))
    else:
        if not os.path.exists(input_path):
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(f"{translations['convert_audio']} '{input_path}'...")
        if os.path.exists(output_path): os.remove(output_path)
        cvt.convert_audio(pitch=pitch, filter_radius=filter_radius, index_rate=index_rate, volume_envelope=volume_envelope, protect=protect, hop_length=hop_length, f0_method=f0_method, audio_input_path=input_path, audio_output_path=output_path, index_path=index_path, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, clean_audio=clean_audio, clean_strength=clean_strength, export_format=export_format, embedder_model=embedder_model, resample_sr=resample_sr, checkpointing=checkpointing, f0_file=f0_file, f0_onnx=f0_onnx, embedders_mode=embedders_mode, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre, split_audio=split_audio)

        if os.path.exists(pid_path): os.remove(pid_path)
        logger.info(translations["convert_audio_success"].format(input_path=input_path, elapsed_time=f"{(time.time() - start_time):.2f}", output_path=output_path.replace('wav', export_format)))

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    rms2 = F.interpolate(torch.from_numpy(librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    return (target_audio * (torch.pow(F.interpolate(torch.from_numpy(librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze(), 1 - rate) * torch.pow(torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6), rate - 1)).numpy())

def get_providers():
    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers

class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs

    def autotune_f0(self, f0, f0_autotune_strength):
        autotuned_f0 = np.zeros_like(f0)

        for i, freq in enumerate(f0):
            autotuned_f0[i] = freq + (min(self.note_dict, key=lambda x: abs(x - freq)) - freq) * f0_autotune_strength

        return autotuned_f0

class VC:
    def __init__(self, tgt_sr, config):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device
        self.is_half = config.is_half
        self.ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00,  207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict

    def get_f0_pm(self, x, p_len):
        import parselmouth

        f0 = (parselmouth.Sound(x, self.sample_rate).to_pitch_ac(time_step=self.window / self.sample_rate * 1000 / 1000, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])
        pad_size = (p_len - len(f0) + 1) // 2

        if pad_size > 0 or p_len - len(f0) - pad_size > 0: f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0
 
    def get_f0_mangio_crepe(self, x, p_len, hop_length, model="full", onnx=False):
        from main.library.predictors.CREPE import predict

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        audio = torch.unsqueeze(torch.from_numpy(x).to(self.device, copy=True), dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True).detach()

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(predict(audio.detach(), self.sample_rate, hop_length, self.f0_min, self.f0_max, model, batch_size=hop_length * 2, device=self.device, pad=True, providers=get_providers(), onnx=onnx).squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan

        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_crepe(self, x, model="full", onnx=False):
        from main.library.predictors.CREPE import predict, mean, median

        f0, pd = predict(torch.tensor(np.copy(x))[None].float(), self.sample_rate, self.window, self.f0_min, self.f0_max, model, batch_size=512, device=self.device, return_periodicity=True, providers=get_providers(), onnx=onnx)
        f0, pd = mean(f0, 3), median(pd, 3)
        f0[pd < 0.1] = 0

        return f0[0].cpu().numpy()

    def get_f0_fcpe(self, x, p_len, hop_length, onnx=False, legacy=False):
        from main.library.predictors.FCPE import FCPE

        model_fcpe = FCPE(os.path.join("assets", "models", "predictors", ("fcpe_legacy" if legacy else"fcpe") + (".onnx" if onnx else ".pt")), hop_length=int(hop_length), f0_min=int(self.f0_min), f0_max=int(self.f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03 if legacy else 0.006, providers=get_providers(), onnx=onnx, legacy=legacy, is_half=self.is_half)
        f0 = model_fcpe.compute_f0(x, p_len=p_len)

        del model_fcpe
        return f0
    
    def get_f0_rmvpe(self, x, legacy=False, onnx=False):
        from main.library.predictors.RMVPE import RMVPE

        rmvpe_model = RMVPE(os.path.join("assets", "models", "predictors", "rmvpe" + (".onnx" if onnx else ".pt")), is_half=self.is_half, device=self.device, onnx=onnx, providers=get_providers())
        f0 = rmvpe_model.infer_from_audio_with_pitch(x, thred=0.03, f0_min=self.f0_min, f0_max=self.f0_max) if legacy else rmvpe_model.infer_from_audio(x, thred=0.03)

        del rmvpe_model
        return f0

    def get_f0_pyworld(self, x, filter_radius, model="harvest"):
        from main.library.predictors.WORLD_WRAPPER import PYWORLD

        pw = PYWORLD()
        x = x.astype(np.double)

        if model == "harvest": f0, t = pw.harvest(x, fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        elif model == "dio": f0, t = pw.dio(x, fs=self.sample_rate, f0_ceil=self.f0_max, f0_floor=self.f0_min, frame_period=10)
        else: raise ValueError(translations["method_not_valid"])

        f0 = pw.stonemask(x, self.sample_rate, t, f0)

        if filter_radius > 2 or model == "dio": f0 = signal.medfilt(f0, filter_radius)
        return f0
    
    def get_f0_swipe(self, x):
        from main.library.predictors.SWIPE import swipe

        f0, _ = swipe(x.astype(np.float32), self.sample_rate, f0_floor=self.f0_min, f0_ceil=self.f0_max, frame_period=10)
        return f0
    
    def get_f0_yin(self, x, hop_length, p_len, mode="yin"):
        if mode == "yin": source = np.array(librosa.yin(x.astype(np.float32), sr=self.sample_rate, fmin=self.f0_min, fmax=self.f0_max, hop_length=hop_length))
        else:
            f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=self.f0_min, fmax=self.f0_max, sr=self.sample_rate, hop_length=hop_length)
            source = np.array(f0)

        source[source < 0.001] = np.nan
        return np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

    def get_f0_hybrid(self, methods_str, x, p_len, hop_length, filter_radius, onnx_mode):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        f0_computation_stack, resampled_stack = [], []
        logger.debug(translations["hybrid_methods"].format(methods=methods))

        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        for method in methods:
            f0 = None
            f0_methods = {"pm": lambda: self.get_f0_pm(x, p_len), "dio": lambda: self.get_f0_pyworld(x, filter_radius, "dio"), "mangio-crepe-tiny": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=onnx_mode), "mangio-crepe-small": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=onnx_mode), "mangio-crepe-medium": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=onnx_mode), "mangio-crepe-large": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=onnx_mode), "mangio-crepe-full": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=onnx_mode), "crepe-tiny": lambda: self.get_f0_crepe(x, "tiny", onnx=onnx_mode), "crepe-small": lambda: self.get_f0_crepe(x, "small", onnx=onnx_mode), "crepe-medium": lambda: self.get_f0_crepe(x, "medium", onnx=onnx_mode), "crepe-large": lambda: self.get_f0_crepe(x, "large", onnx=onnx_mode), "crepe-full": lambda: self.get_f0_crepe(x, "full", onnx=onnx_mode), "fcpe": lambda: self.get_f0_fcpe(x, p_len, int(hop_length), onnx=onnx_mode), "fcpe-legacy": lambda: self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True, onnx=onnx_mode), "rmvpe": lambda: self.get_f0_rmvpe(x, onnx=onnx_mode), "rmvpe-legacy": lambda: self.get_f0_rmvpe(x, legacy=True, onnx=onnx_mode), "harvest": lambda: self.get_f0_pyworld(x, filter_radius, "harvest"), "yin": lambda: self.get_f0_yin(x, int(hop_length), p_len, mode="yin"), "pyin": lambda: self.get_f0_yin(x, int(hop_length), p_len, mode="pyin"), "swipe": lambda: self.get_f0_swipe(x)}
            f0 = f0_methods.get(method, lambda: ValueError(translations["method_not_valid"]))()
            f0_computation_stack.append(f0) 

        for f0 in f0_computation_stack:
            resampled_stack.append(np.interp(np.linspace(0, len(f0), p_len), np.arange(len(f0)), f0))

        return resampled_stack[0] if len(resampled_stack) == 1 else np.nanmedian(np.vstack(resampled_stack), axis=0)

    def get_f0(self, x, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength, inp_f0=None, onnx_mode=False):
        f0_methods = {"pm": lambda: self.get_f0_pm(x, p_len), "dio": lambda: self.get_f0_pyworld(x, filter_radius, "dio"), "mangio-crepe-tiny": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "tiny", onnx=onnx_mode), "mangio-crepe-small": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "small", onnx=onnx_mode), "mangio-crepe-medium": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "medium", onnx=onnx_mode), "mangio-crepe-large": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "large", onnx=onnx_mode), "mangio-crepe-full": lambda: self.get_f0_mangio_crepe(x, p_len, int(hop_length), "full", onnx=onnx_mode), "crepe-tiny": lambda: self.get_f0_crepe(x, "tiny", onnx=onnx_mode), "crepe-small": lambda: self.get_f0_crepe(x, "small", onnx=onnx_mode), "crepe-medium": lambda: self.get_f0_crepe(x, "medium", onnx=onnx_mode), "crepe-large": lambda: self.get_f0_crepe(x, "large", onnx=onnx_mode), "crepe-full": lambda: self.get_f0_crepe(x, "full", onnx=onnx_mode), "fcpe": lambda: self.get_f0_fcpe(x, p_len, int(hop_length), onnx=onnx_mode), "fcpe-legacy": lambda: self.get_f0_fcpe(x, p_len, int(hop_length), legacy=True, onnx=onnx_mode), "rmvpe": lambda: self.get_f0_rmvpe(x, onnx=onnx_mode), "rmvpe-legacy": lambda: self.get_f0_rmvpe(x, legacy=True, onnx=onnx_mode), "harvest": lambda: self.get_f0_pyworld(x, filter_radius, "harvest"), "yin": lambda: self.get_f0_yin(x, int(hop_length), p_len, mode="yin"), "pyin": lambda: self.get_f0_yin(x, int(hop_length), p_len, mode="pyin"), "swipe": lambda: self.get_f0_swipe(x)}
        f0 = self.get_f0_hybrid(f0_method, x, p_len, hop_length, filter_radius, onnx_mode) if "hybrid" in f0_method else f0_methods.get(f0_method, lambda: ValueError(translations["method_not_valid"]))()

        if f0_autotune: f0 = Autotune.autotune_f0(self, f0, f0_autotune_strength)
        if isinstance(f0, tuple): f0 = f0[0]

        f0 *= pow(2, pitch / 12)
        tf0 = self.sample_rate // self.window

        if inp_f0 is not None:
            replace_f0 = np.interp(list(range(np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype(np.int16))), inp_f0[:, 0] * 100, inp_f0[:, 1])
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]]

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255

        return np.rint(f0_mel).astype(np.int32), f0.copy()
    
    def extract_features(self, model, feats, version):
        return torch.as_tensor(model.run([model.get_outputs()[0].name, model.get_outputs()[1].name], {"feats": feats.detach().cpu().numpy()})[0 if version == "v1" else 1], dtype=torch.float32, device=feats.device)

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        pitch_guidance = pitch != None and pitchf != None
        feats = (torch.from_numpy(audio0).half() if self.is_half else torch.from_numpy(audio0).float())

        if feats.dim() == 2: feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        feats = feats.view(1, -1)

        with torch.no_grad():
            if self.embed_suffix == ".pt":
                padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
                logits = model.extract_features(**{"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12})
                feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
            elif self.embed_suffix == ".onnx": feats = self.extract_features(model, feats.to(self.device), version).to(self.device)
            elif self.embed_suffix == ".safetensors":
                logits = model(feats.to(self.device))["last_hidden_state"]
                feats = (model.final_proj(logits[0]).unsqueeze(0) if version == "v1" else logits)
            else: raise ValueError(translations["option_not_valid"])

            if protect < 0.5 and pitch_guidance: feats0 = feats.clone()

            if (not isinstance(index, type(None)) and not isinstance(big_npy, type(None)) and index_rate != 0):
                npy = feats[0].cpu().numpy()
                if self.is_half: npy = npy.astype(np.float32)

                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)

                npy = np.sum(big_npy[ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
                if self.is_half: npy = npy.astype(np.float16)

                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if protect < 0.5 and pitch_guidance: feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            p_len = audio0.shape[0] // self.window
            
            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch_guidance:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if protect < 0.5 and pitch_guidance:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            audio1 = ((net_g.infer(feats.half() if self.is_half else feats.float(), p_len, pitch if pitch_guidance else None, (pitchf.half() if self.is_half else pitchf.float()) if pitch_guidance else None, sid)[0][0, 0]).data.cpu().float().numpy()) if self.suffix == ".pth" else (net_g.run([net_g.get_outputs()[0].name], ({net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32), net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64), net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32)} if pitch_guidance else {net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32), net_g.get_inputs()[1].name: p_len.cpu().numpy(), net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64), net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32)}))[0][0, 0])

        if self.embed_suffix == ".pt": del padding_mask
        del feats, p_len, net_g

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        return audio1
    
    def pipeline(self, model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, filter_radius, volume_envelope, version, protect, hop_length, f0_autotune, f0_autotune_strength, suffix, embed_suffix, f0_file=None, f0_onnx=False, pbar=None):
        self.suffix = suffix
        self.embed_suffix = embed_suffix

        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                logger.error(translations["read_faiss_index_error"].format(e=e))
                index = big_npy = None
        else: index = big_npy = None

        pbar.update(1)
        opt_ts, audio_opt = [], []
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])

        s = 0
        t, inp_f0 = None, None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    raw_lines = f.read()
                    if len(raw_lines) > 0:
                        inp_f0 = []
                        for line in raw_lines.strip("\n").split("\n"):
                            inp_f0.append([float(i) for i in line.split(",")])

                        inp_f0 = np.array(inp_f0, dtype=np.float32)
            except:
                logger.error(translations["error_readfile"])
                inp_f0 = None

        pbar.update(1)
        if pitch_guidance:
            pitch, pitchf = self.get_f0(audio_pad, p_len, pitch, f0_method, filter_radius, hop_length, f0_autotune, f0_autotune_strength, inp_f0, onnx_mode=f0_onnx)
            pitch, pitchf = pitch[:p_len], pitchf[:p_len]
            if self.device == "mps": pitchf = pitchf.astype(np.float32)
            pitch, pitchf = torch.tensor(pitch, device=self.device).unsqueeze(0).long(), torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        pbar.update(1)
        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])    
            s = t
            
        audio_opt.append(self.voice_conversion(model, net_g, sid, audio_pad[t:], (pitch[:, t // self.window :] if t is not None else pitch) if pitch_guidance else None, (pitchf[:, t // self.window :] if t is not None else pitchf) if pitch_guidance else None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: audio_opt = change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope)
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max
        if pitch_guidance: del pitch, pitchf
        del sid

        pbar.update(1)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        return audio_opt

class VoiceConverter:
    def __init__(self, model_path, sid = 0):
        self.config = config
        self.device = config.device
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"
        self.checkpointing = False
        self.sid = sid
        self.get_vc(model_path, sid)

    def convert_audio(self, audio_input_path, audio_output_path, index_path, embedder_model, pitch, f0_method, index_rate, volume_envelope, protect, hop_length, f0_autotune, f0_autotune_strength, filter_radius, clean_audio, clean_strength, export_format, resample_sr = 0, checkpointing = False, f0_file = None, f0_onnx = False, embedders_mode = "fairseq", formant_shifting = False, formant_qfrency = 0.8, formant_timbre = 0.8, split_audio = False):
        try:
            with tqdm(total=10, desc=translations["convert_audio"], ncols=100, unit="a") as pbar:
                audio = load_audio(logger, audio_input_path, 16000, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
                self.checkpointing = checkpointing
                audio_max = np.abs(audio).max() / 0.95
                if audio_max > 1: audio /= audio_max

                pbar.update(1)
                if not self.hubert_model:
                    models, _, embed_suffix = load_embedders_model(embedder_model, embedders_mode, providers=get_providers())
                    self.hubert_model = (models.to(self.device).half() if self.config.is_half else models.to(self.device).float()).eval() if embed_suffix in [".pt", ".safetensors"] else models
                    self.embed_suffix = embed_suffix

                pbar.update(1)
                if self.tgt_sr != resample_sr >= 16000: self.tgt_sr = resample_sr
                target_sr = min([8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000], key=lambda x: abs(x - self.tgt_sr))

                if split_audio:
                    chunks = cut(audio, 16000, db_thresh=-60, min_interval=500)  
                    pbar.total = len(chunks) * 4 + 6
                    logger.info(f"{translations['split_total']}: {len(chunks)}")
                else: chunks = [(audio, 0, 0)]

                converted_chunks = []
                pbar.update(1)

                for waveform, start, end in chunks:
                    converted_chunks.append((start, end, self.vc.pipeline(model=self.hubert_model, net_g=self.net_g, sid=self.sid, audio=waveform, pitch=pitch, f0_method=f0_method, file_index=(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")), index_rate=index_rate, pitch_guidance=self.use_f0, filter_radius=filter_radius, volume_envelope=volume_envelope, version=self.version, protect=protect, hop_length=hop_length, f0_autotune=f0_autotune, f0_autotune_strength=f0_autotune_strength, suffix=self.suffix, embed_suffix=self.embed_suffix, f0_file=f0_file, f0_onnx=f0_onnx, pbar=pbar)))
                
                pbar.update(1)
                audio_output = restore(converted_chunks, total_len=len(audio), dtype=converted_chunks[0][2].dtype) if split_audio else converted_chunks[0][2]
                if target_sr >= 16000 and self.tgt_sr != target_sr: audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=target_sr, res_type="soxr_vhq")

                pbar.update(1)
                if clean_audio:
                    from main.tools.noisereduce import reduce_noise
                    audio_output = reduce_noise(y=audio_output, sr=target_sr, prop_decrease=clean_strength, device=self.device) 

                sf.write(audio_output_path, audio_output, target_sr, format=export_format)
                pbar.update(1)
        except Exception as e:
            logger.error(translations["error_convert"].format(e=e))
            import traceback
            logger.debug(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            self.load_model()
            if self.cpt is not None: self.setup()

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        del self.net_g, self.cpt

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        self.cpt = None

    def load_model(self):
        if os.path.isfile(self.loaded_model):
            if self.loaded_model.endswith(".pth"): self.cpt = torch.load(self.loaded_model, map_location="cpu")  
            else: 
                sess_options = onnxruntime.SessionOptions()
                sess_options.log_severity_level = 3
                self.cpt = onnxruntime.InferenceSession(self.loaded_model, sess_options=sess_options, providers=get_providers())
        else: self.cpt = None

    def setup(self):
        if self.cpt is not None:
            if self.loaded_model.endswith(".pth"):
                self.tgt_sr = self.cpt["config"][-1]
                self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
                self.use_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                self.vocoder = self.cpt.get("vocoder", "Default")

                self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=768 if self.version == "v2" else 256, vocoder=self.vocoder, checkpointing=self.checkpointing)
                del self.net_g.enc_q

                self.net_g.load_state_dict(self.cpt["weight"], strict=False)
                self.net_g.eval().to(self.device)
                self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())

                self.n_spk = self.cpt["config"][-3]
                self.suffix = ".pth"
            else:
                import json
                import onnx

                metadata_dict = None
                for prop in onnx.load(self.loaded_model).metadata_props:
                    if prop.key == "model_info":
                        metadata_dict = json.loads(prop.value)
                        break

                self.net_g = self.cpt
                self.tgt_sr = metadata_dict.get("sr", 32000)
                self.use_f0 = metadata_dict.get("f0", 1)
                self.version = metadata_dict.get("version", "v1")
                self.suffix = ".onnx"

            self.vc = VC(self.tgt_sr, self.config)

if __name__ == "__main__": main()