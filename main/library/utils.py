import os
import re
import sys
import codecs
import librosa
import logging

import numpy as np
import soundfile as sf

from pydub import AudioSegment, silence

sys.path.append(os.getcwd())

from main.tools import huggingface
from main.configs.config import Config

for l in ["httpx", "httpcore"]:
    logging.getLogger(l).setLevel(logging.ERROR)

translations = Config().translations

def check_predictors(method, f0_onnx=False):
    if f0_onnx and method not in ["harvestw", "diow"]: method += "-onnx"

    def download(predictors):
        if not os.path.exists(os.path.join("assets", "models", "predictors", predictors)): huggingface.HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, os.path.join("assets", "models", "predictors", predictors))

    model_dict = {**dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), **dict.fromkeys(["rmvpe-onnx", "rmvpe-legacy-onnx"], "rmvpe.onnx"), **dict.fromkeys(["fcpe"], "fcpe.pt"), **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), **dict.fromkeys(["fcpe-onnx"], "fcpe.onnx"), **dict.fromkeys(["fcpe-legacy-onnx"], "fcpe_legacy.onnx"), **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), **dict.fromkeys(["crepe-full-onnx", "mangio-crepe-full-onnx"], "crepe_full.onnx"), **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), **dict.fromkeys(["crepe-large-onnx", "mangio-crepe-large-onnx"], "crepe_large.onnx"), **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), **dict.fromkeys(["crepe-medium-onnx", "mangio-crepe-medium-onnx"], "crepe_medium.onnx"), **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), **dict.fromkeys(["crepe-small-onnx", "mangio-crepe-small-onnx"], "crepe_small.onnx"), **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), **dict.fromkeys(["crepe-tiny-onnx", "mangio-crepe-tiny-onnx"], "crepe_tiny.onnx"), **dict.fromkeys(["harvestw", "diow"], "world.pth")}

    if "hybrid" in method:
        methods_str = re.search("hybrid\[(.+)\]", method)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        for method in methods:
            if method in model_dict: download(model_dict[method])
    elif method in model_dict: download(model_dict[method])

def check_embedders(hubert, embedders_mode="fairseq"):
    huggingface_url = codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/rzorqqref/", "rot13")

    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", "korean_hubert_base", "chinese_hubert_base", "portuguese_hubert_base"]:
        if embedders_mode == "fairseq": hubert += ".pt"
        elif embedders_mode == "onnx": hubert += ".onnx"

        model_path = os.path.join("assets", "models", "embedders", hubert)

        if embedders_mode == "fairseq": 
            if not os.path.exists(model_path): huggingface.HF_download_file("".join([huggingface_url, "fairseq/", hubert]), model_path)
        elif embedders_mode == "onnx": 
            if not os.path.exists(model_path): huggingface.HF_download_file("".join([huggingface_url, "onnx/", hubert]), model_path)
        elif embedders_mode == "transformers":
            bin_file = os.path.join(model_path, "pytorch_model.bin")
            config_file = os.path.join(model_path, "config.json")

            os.makedirs(model_path, exist_ok=True)

            if not os.path.exists(bin_file): huggingface.HF_download_file("".join([huggingface_url, "transformers/", hubert, "/pytorch_model.bin"]), bin_file)
            if not os.path.exists(config_file): huggingface.HF_download_file("".join([huggingface_url, "transformers/", hubert, "/config.json"]), config_file)
        else: raise ValueError(translations["option_not_valid"])
    
def load_audio(logger, file, sample_rate=16000, formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file): raise FileNotFoundError(translations["not_found"].format(name=file))

        try:
            logger.debug(translations['read_sf'])
            audio, sr = sf.read(file)
        except:
            logger.debug(translations['read_librosa'])
            audio, sr = librosa.load(file, sr=None)

        if len(audio.shape) > 1: audio = librosa.to_mono(audio.T)
        if sr != sample_rate: audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_vhq")

        if formant_shifting:
            from main.library.algorithm.stftpitchshift import StftPitchShift

            pitchshifter = StftPitchShift(1024, 32, sample_rate)
            audio = pitchshifter.shiftpitch(audio, factors=1, quefrency=formant_qfrency * 1e-3, distortion=formant_timbre)
    except Exception as e:
        raise RuntimeError(f"{translations['errors_loading_audio']}: {e}")
    
    return audio.flatten()

def process_audio(logger, file_path, output_path):
    try:
        song = pydub_convert(pydub_load(file_path))
        cut_files, time_stamps = [], []

        for i, (start_i, end_i) in enumerate(silence.detect_nonsilent(song, min_silence_len=250, silence_thresh=-60)):
            chunk = song[start_i:end_i]

            chunk_file_path = os.path.join(output_path, f"chunk{i}.wav")
            logger.debug(f"{chunk_file_path}: {len(chunk)}")

            if os.path.exists(chunk_file_path): os.remove(chunk_file_path)
            chunk.export(chunk_file_path, format="wav")

            cut_files.append(chunk_file_path)
            time_stamps.append((start_i, end_i))
        
        logger.info(f"{translations['split_total']}: {len(cut_files)}")
        return cut_files, time_stamps
    except Exception as e:
        raise RuntimeError(f"{translations['process_audio_error']}: {e}")

def merge_audio(files_list, time_stamps, original_file_path, output_path, format):
    try:
        def extract_number(filename):
            match = re.search(r'_(\d+)', filename)
            return int(match.group(1)) if match else 0

        total_duration = len(pydub_load(original_file_path))

        combined = AudioSegment.empty() 
        current_position = 0 

        for file, (start_i, end_i) in zip(sorted(files_list, key=extract_number), time_stamps):
            if start_i > current_position: combined += AudioSegment.silent(duration=start_i - current_position)  
            
            combined += pydub_load(file)  
            current_position = end_i

        if current_position < total_duration: combined += AudioSegment.silent(duration=total_duration - current_position)
        combined.export(output_path, format=format)

        return output_path
    except Exception as e:
        raise RuntimeError(f"{translations['merge_error']}: {e}")

def pydub_convert(audio):
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    if samples.dtype != np.int16: samples = (samples * 32767).astype(np.int16)

    return AudioSegment(samples.tobytes(), frame_rate=audio.frame_rate, sample_width=samples.dtype.itemsize, channels=audio.channels)

def pydub_load(input_path):
    if input_path.endswith(".wav"): audio = AudioSegment.from_wav(input_path)
    elif input_path.endswith(".mp3"): audio = AudioSegment.from_mp3(input_path)
    elif input_path.endswith(".ogg"): audio = AudioSegment.from_ogg(input_path)
    else: audio = AudioSegment.from_file(input_path)
    
    return audio

def load_embedders_model(embedder_model, embedders_mode="fairseq", providers=None):
    if embedders_mode == "fairseq": embedder_model += ".pt"
    elif embedders_mode == "onnx": embedder_model += ".onnx"

    embedder_model_path = os.path.join("assets", "models", "embedders", embedder_model)
    if not os.path.exists(embedder_model_path): raise FileNotFoundError(f"{translations['not_found'].format(name=translations['model'])}: {embedder_model}")

    try:
        if embedders_mode == "fairseq":
            from fairseq import checkpoint_utils

            models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task([embedder_model_path], suffix="")
            embed_suffix = ".pt"

            hubert_model = models[0]
        elif embedders_mode == "onnx":
            import onnxruntime

            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 3

            embed_suffix, saved_cfg = ".onnx", None
            hubert_model = onnxruntime.InferenceSession(embedder_model_path, sess_options=sess_options, providers=providers)
        elif embedders_mode == "transformers":
            from torch import nn
            from transformers import HubertModel

            class HubertModelWithFinalProj(HubertModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
                    
            embed_suffix, saved_cfg = ".bin", None
            hubert_model = HubertModelWithFinalProj.from_pretrained(embedder_model_path)
        else: raise ValueError(translations["option_not_valid"])
    except Exception as e:
        raise RuntimeError(translations["read_model_error"].format(e=e))
    
    return hubert_model, saved_cfg, embed_suffix