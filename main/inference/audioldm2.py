import os
import sys
import time
import tqdm
import torch
import logging
import librosa
import argparse
import scipy.signal
import logging.handlers

import numpy as np
import soundfile as sf

from torch import inference_mode
from distutils.util import strtobool

sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.audioldm2.utils import load_audio
from main.library.audioldm2.models import load_model

config = Config()
translations = config.translations
logger = logging.getLogger(__name__)
logger.propagate = False

for l in ["torch", "httpx", "httpcore", "diffusers", "transformers"]:
    logging.getLogger(l).setLevel(logging.ERROR)

if logger.hasHandlers(): logger.handlers.clear()
else:
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(os.path.join("assets", "logs", "audioldm2.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_formatter = logging.Formatter(fmt="\n%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./output.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--audioldm_model", type=str, default="audioldm2-music")
    parser.add_argument("--source_prompt", type=str, default="")
    parser.add_argument("--target_prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cfg_scale_src", type=float, default=3.5)
    parser.add_argument("--cfg_scale_tar", type=float, default=12)
    parser.add_argument("--t_start", type=int, default=45)
    parser.add_argument("--save_compute", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def main():
    args = parse_arguments()
    input_path, output_path, export_format, sample_rate, audioldm_model, source_prompt, target_prompt, steps, cfg_scale_src, cfg_scale_tar, t_start, save_compute = args.input_path, args.output_path, args.export_format, args.sample_rate, args.audioldm_model, args.source_prompt, args.target_prompt, args.steps, args.cfg_scale_src, args.cfg_scale_tar, args.t_start, args.save_compute

    log_data = {translations['audio_path']: input_path, translations['output_path']: output_path.replace('wav', export_format), translations['model_name']: audioldm_model, translations['export_format']: export_format, translations['sample_rate']: sample_rate, translations['steps']: steps, translations['source_prompt']: source_prompt, translations['target_prompt']: target_prompt, translations['cfg_scale_src']: cfg_scale_src, translations['cfg_scale_tar']: cfg_scale_tar, translations['t_start']: t_start, translations['save_compute']: save_compute}

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")
   
    start_time = time.time()
    logger.info(translations["start_edit"].format(input_path=input_path))
    
    try:
        edit(input_path, output_path, audioldm_model, source_prompt, target_prompt, steps, cfg_scale_src, cfg_scale_tar, t_start, save_compute, sample_rate, config.device, export_format=export_format)
    except Exception as e:
        logger.error(translations["error_edit"].format(e=e))

        import traceback
        logger.debug(traceback.format_exc())
        
    logger.info(translations["edit_success"].format(time=f"{(time.time() - start_time):.2f}", output_path=output_path.replace('wav', export_format)))

def invert(ldm_stable, x0, prompt_src, num_diffusion_steps, cfg_scale_src, duration, save_compute):
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    _, zs, wts, extra_info = inversion_forward_process(ldm_stable, w0, etas=1, prompts=[prompt_src], cfg_scales=[cfg_scale_src], num_inference_steps=num_diffusion_steps, numerical_fix=True, duration=duration, save_compute=save_compute)
    return zs, wts, extra_info

def low_pass_filter(audio, cutoff=7500, sr=16000):
    b, a = scipy.signal.butter(4, cutoff / (sr / 2), btype='low')
    return scipy.signal.filtfilt(b, a, audio)

def sample(output_audio, sr, ldm_stable, zs, wts, extra_info, prompt_tar, tstart, cfg_scale_tar, duration, save_compute, export_format = "wav"):
    tstart = torch.tensor(tstart, dtype=torch.int32)
    w0, _ = inversion_reverse_process(ldm_stable, xT=wts, tstart=tstart, etas=1., prompts=[prompt_tar], neg_prompts=[""], cfg_scales=[cfg_scale_tar], zs=zs[:int(tstart)], duration=duration, extra_info=extra_info, save_compute=save_compute)

    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)

    if x0_dec.dim() < 4: x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)

    audio = audio.squeeze().cpu().numpy()
    orig_sr = ldm_stable.get_sr()

    if sr != 16000 and sr > 0: 
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr, res_type="soxr_vhq")
        orig_sr = sr

    audio = low_pass_filter(audio, 7500, orig_sr)

    sf.write(output_audio, np.tile(audio, (2, 1)).T, orig_sr, format=export_format)
    return output_audio

def edit(input_audio, output_audio, model_id, source_prompt = "", target_prompt = "", steps = 200, cfg_scale_src = 3.5, cfg_scale_tar = 12, t_start = 45, save_compute = True, sr = 44100, device = "cpu", export_format = "wav"):
    ldm_stable = load_model(model_id, device=device)
    ldm_stable.model.scheduler.set_timesteps(steps, device=device)

    x0, duration = load_audio(input_audio, ldm_stable.get_melspectrogram(), device=device)
    zs_tensor, wts_tensor, extra_info_list = invert(ldm_stable=ldm_stable, x0=x0, prompt_src=source_prompt, num_diffusion_steps=steps, cfg_scale_src=cfg_scale_src, duration=duration, save_compute=save_compute)

    return sample(output_audio, sr, ldm_stable, zs_tensor, wts_tensor, extra_info_list, prompt_tar=target_prompt, tstart=int(t_start / 100 * steps), cfg_scale_tar=cfg_scale_tar, duration=duration, save_compute=save_compute, export_format=export_format)

def inversion_forward_process(model, x0, etas = None, prompts = [""], cfg_scales = [3.5], num_inference_steps = 50, numerical_fix = False, duration = None, first_order = False, save_compute = True):
    if len(prompts) > 1 or prompts[0] != "":
        text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = model.encode_text(prompts)
        uncond_embeddings_hidden_states, uncond_embeddings_class_lables, uncond_boolean_prompt_mask = model.encode_text([""], negative=True, save_compute=save_compute, cond_length=text_embeddings_class_labels.shape[1] if text_embeddings_class_labels is not None else None)
    else: uncond_embeddings_hidden_states, uncond_embeddings_class_lables, uncond_boolean_prompt_mask = model.encode_text([""], negative=True, save_compute=False)

    timesteps = model.model.scheduler.timesteps.to(model.device)
    variance_noise_shape = model.get_noise_shape(x0, num_inference_steps)

    if type(etas) in [int, float]: etas = [etas]*model.model.scheduler.num_inference_steps

    xts = model.sample_xts_from_x0(x0, num_inference_steps=num_inference_steps)
    zs = torch.zeros(size=variance_noise_shape, device=model.device)
    extra_info = [None] * len(zs)

    if timesteps[0].dtype == torch.int64: t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    elif timesteps[0].dtype == torch.float32: t_to_idx = {float(v): k for k, v in enumerate(timesteps)}

    xt = x0
    model.setup_extra_inputs(xt, init_timestep=timesteps[0], audio_end_in_s=duration, save_compute=save_compute and prompts[0] != "")

    for t in tqdm.tqdm(timesteps, desc=translations["inverting"], ncols=100, unit="a"):
        idx = num_inference_steps - t_to_idx[int(t) if timesteps[0].dtype == torch.int64 else float(t)] - 1
        xt = xts[idx + 1][None]
        xt_inp = model.model.scheduler.scale_model_input(xt, t)

        with torch.no_grad():
            if save_compute and prompts[0] != "":
                comb_out, _, _ = model.unet_forward(xt_inp.expand(2, -1, -1, -1) if hasattr(model.model, 'unet') else xt_inp.expand(2, -1, -1), timestep=t, encoder_hidden_states=torch.cat([uncond_embeddings_hidden_states, text_embeddings_hidden_states], dim=0) if uncond_embeddings_hidden_states is not None else None, class_labels=torch.cat([uncond_embeddings_class_lables, text_embeddings_class_labels], dim=0) if uncond_embeddings_class_lables is not None else None, encoder_attention_mask=torch.cat([uncond_boolean_prompt_mask, text_embeddings_boolean_prompt_mask], dim=0) if uncond_boolean_prompt_mask is not None else None)
                out, cond_out = comb_out.sample.chunk(2, dim=0)
            else:
                out = model.unet_forward(xt_inp, timestep=t, encoder_hidden_states=uncond_embeddings_hidden_states, class_labels=uncond_embeddings_class_lables, encoder_attention_mask=uncond_boolean_prompt_mask)[0].sample
                if len(prompts) > 1 or prompts[0] != "": cond_out = model.unet_forward(xt_inp, timestep=t, encoder_hidden_states=text_embeddings_hidden_states, class_labels=text_embeddings_class_labels, encoder_attention_mask=text_embeddings_boolean_prompt_mask)[0].sample

        if len(prompts) > 1 or prompts[0] != "": noise_pred = out + (cfg_scales[0] * (cond_out - out)).sum(axis=0).unsqueeze(0)
        else: noise_pred = out

        xtm1 = xts[idx][None]
        z, xtm1, extra = model.get_zs_from_xts(xt, xtm1, noise_pred, t, eta=etas[idx], numerical_fix=numerical_fix, first_order=first_order)

        zs[idx] = z
        xts[idx] = xtm1
        extra_info[idx] = extra

    if zs is not None: zs[0] = torch.zeros_like(zs[0])
    return xt, zs, xts, extra_info

def inversion_reverse_process(model, xT, tstart, etas = 0, prompts = [""], neg_prompts = [""], cfg_scales = None, zs = None, duration = None, first_order = False, extra_info = None, save_compute = True):
    text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = model.encode_text(prompts)
    uncond_embeddings_hidden_states, uncond_embeddings_class_lables, uncond_boolean_prompt_mask = model.encode_text(neg_prompts, negative=True, save_compute=save_compute, cond_length=text_embeddings_class_labels.shape[1] if text_embeddings_class_labels is not None else None)
    xt = xT[tstart.max()].unsqueeze(0)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.model.scheduler.num_inference_steps
    
    assert len(etas) == model.model.scheduler.num_inference_steps
    timesteps = model.model.scheduler.timesteps.to(model.device)

    if timesteps[0].dtype == torch.int64: t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    elif timesteps[0].dtype == torch.float32: t_to_idx = {float(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    model.setup_extra_inputs(xt, extra_info=extra_info, init_timestep=timesteps[-zs.shape[0]], audio_end_in_s=duration, save_compute=save_compute)

    for t in tqdm.tqdm(timesteps[-zs.shape[0]:], desc=translations["editing"], ncols=100, unit="a"):
        idx = model.model.scheduler.num_inference_steps - t_to_idx[int(t) if timesteps[0].dtype == torch.int64 else float(t)] - (model.model.scheduler.num_inference_steps - zs.shape[0] + 1)
        xt_inp = model.model.scheduler.scale_model_input(xt, t)

        with torch.no_grad():
            if save_compute:
                comb_out, _, _ = model.unet_forward(xt_inp.expand(2, -1, -1, -1) if hasattr(model.model, 'unet') else xt_inp.expand(2, -1, -1), timestep=t, encoder_hidden_states=torch.cat([uncond_embeddings_hidden_states, text_embeddings_hidden_states], dim=0) if uncond_embeddings_hidden_states is not None else None, class_labels=torch.cat([uncond_embeddings_class_lables, text_embeddings_class_labels], dim=0) if uncond_embeddings_class_lables is not None else None, encoder_attention_mask=torch.cat([uncond_boolean_prompt_mask, text_embeddings_boolean_prompt_mask], dim=0) if uncond_boolean_prompt_mask is not None else None)
                uncond_out, cond_out = comb_out.sample.chunk(2, dim=0)
            else:
                uncond_out = model.unet_forward(xt_inp, timestep=t, encoder_hidden_states=uncond_embeddings_hidden_states, class_labels=uncond_embeddings_class_lables, encoder_attention_mask=uncond_boolean_prompt_mask)[0].sample
                cond_out = model.unet_forward(xt_inp, timestep=t, encoder_hidden_states=text_embeddings_hidden_states, class_labels=text_embeddings_class_labels, encoder_attention_mask=text_embeddings_boolean_prompt_mask)[0].sample

        z = zs[idx] if zs is not None else None
        noise_pred = uncond_out + (cfg_scales[0] * (cond_out - uncond_out)).sum(axis=0).unsqueeze(0)
        xt = model.reverse_step_with_custom_noise(noise_pred, t, xt, variance_noise=z.unsqueeze(0), eta=etas[idx], first_order=first_order)

    return xt, zs

if __name__ == "__main__": main()