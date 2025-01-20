import os
import re
import sys
import codecs
import librosa
import logging

import soundfile as sf

from pydub import AudioSegment, silence

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.configs.config import Config

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

translations = Config().translations

def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join("assets", "models", "predictors", predictors)): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, os.path.join("assets", "models", "predictors", predictors))

    if method in ["rmvpe", "rmvpe-legacy"]: download("rmvpe.pt")
    elif method in ["rmvpe-onnx", "rmvpe-legacy-onnx"]: download("rmvpe.onnx")
    elif method in ["fcpe", "fcpe-legacy"]: download("fcpe.pt")
    elif method in ["fcpe-onnx", "fcpe-legacy-onnx"]: download("fcpe.onnx")
    elif method in ["crepe-full", "mangio-crepe-full"]: download("crepe_full.pth")
    elif method in ["crepe-full-onnx", "mangio-crepe-full-onnx"]: download("crepe_full.onnx")
    elif method in ["crepe-large", "mangio-crepe-large"]: download("crepe_large.pth")
    elif method in ["crepe-large-onnx", "mangio-crepe-large-onnx"]: download("crepe_large.onnx")
    elif method in ["crepe-medium", "mangio-crepe-medium"]: download("crepe_medium.pth")
    elif method in ["crepe-medium-onnx", "mangio-crepe-medium-onnx"]: download("crepe_medium.onnx")
    elif method in ["crepe-small", "mangio-crepe-small"]: download("crepe_small.pth")
    elif method in ["crepe-small-onnx", "mangio-crepe-small-onnx"]: download("crepe_small.onnx")
    elif method in ["crepe-tiny", "mangio-crepe-tiny"]: download("crepe_tiny.pth")
    elif method in ["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"]: download("crepe_tiny.onnx")
    elif method in ["harvest", "dio"]: download("world.pth")
    elif "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]
        for method in methods:
            if method in ["rmvpe", "rmvpe-legacy"]: download("rmvpe.pt")
            elif method in ["rmvpe-onnx", "rmvpe-legacy-onnx"]: download("rmvpe.onnx")
            elif method in ["fcpe", "fcpe-legacy"]: download("fcpe.pt")
            elif method in ["fcpe-onnx", "fcpe-legacy-onnx"]: download("fcpe.onnx")
            elif method in ["crepe-full", "mangio-crepe-full"]: download("crepe_full.pth")
            elif method in ["crepe-full-onnx", "mangio-crepe-full-onnx"]: download("crepe_full.onnx")
            elif method in ["crepe-large", "mangio-crepe-large"]: download("crepe_large.pth")
            elif method in ["crepe-large-onnx", "mangio-crepe-large-onnx"]: download("crepe_large.onnx")
            elif method in ["crepe-medium", "mangio-crepe-medium"]: download("crepe_medium.pth")
            elif method in ["crepe-medium-onnx", "mangio-crepe-medium-onnx"]: download("crepe_medium.onnx")
            elif method in ["crepe-small", "mangio-crepe-small"]: download("crepe_small.pth")
            elif method in ["crepe-small-onnx", "mangio-crepe-small-onnx"]: download("crepe_small.onnx")
            elif method in ["crepe-tiny", "mangio-crepe-tiny"]: download("crepe_tiny.pth")
            elif method in ["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"]: download("crepe_tiny.onnx")
            elif method in ["harvest", "dio"]: download("world.pth")

def check_embedders(hubert):
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "Hidden_Rabbit_last", "portuguese_hubert_base"]:
        model_path = os.path.join("assets", "models", "embedders", hubert + '.pt')
        if not os.path.exists(model_path): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13") + f"{hubert}.pt", model_path)

def load_audio(file):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        audio, sr = sf.read(file)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    return audio.flatten()

def process_audio(logger, file_path, output_path):
    try:
        song = AudioSegment.from_file(file_path)
        cut_files, time_stamps = [], []

        for i, (start_i, end_i) in enumerate(silence.detect_nonsilent(song, min_silence_len=750, silence_thresh=-70)):
            chunk = song[start_i:end_i]
            if len(chunk) > 10:
                chunk_file_path = os.path.join(output_path, f"chunk{i}.wav")
                if os.path.exists(chunk_file_path): os.remove(chunk_file_path)

                chunk.export(chunk_file_path, format="wav")
                cut_files.append(chunk_file_path)
                time_stamps.append((start_i, end_i))
            else: logger.debug(translations["skip_file"].format(i=i, chunk=len(chunk)))

        logger.info(f"{translations['split_total']}: {len(cut_files)}")
        return cut_files, time_stamps
    except Exception as e:
        raise RuntimeError(f"{translations['process_audio_error']}: {e}")

def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
    try:
        def extract_number(filename):
            match = re.search(r'_(\d+)', filename)
            return int(match.group(1)) if match else 0

        total_duration = len(AudioSegment.from_file(original_file_path))
        combined = AudioSegment.empty() 
        current_position = 0 

        for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
            if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
            combined += AudioSegment.from_file(file)  
            current_position = end_i

        if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
        combined.export(output_path, format=format)
        return output_path
    except Exception as e:
        raise RuntimeError(f"{translations['merge_error']}: {e}")