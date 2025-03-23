import os
import sys
import torch
import librosa

import numpy as np
import torch.nn.functional as F

from scipy.signal import get_window
from librosa.util import pad_center
from diffusers import DDIMScheduler, AudioLDM2Pipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers import RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer

sys.path.append(os.getcwd())

from main.configs.config import Config
from main.library.utils import check_audioldm2

config = Config()

class Pipeline(torch.nn.Module):
    def __init__(self, model_id, device, double_precision = False, token = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.device = device
        self.double_precision = double_precision
        self.token = token

    def load_scheduler(self):
        pass

    def get_melspectrogram(self):
        pass

    def vae_encode(self, x):
        pass

    def vae_decode(self, x):
        pass

    def decode_to_mel(self, x):
        pass

    def setup_extra_inputs(self, *args, **kwargs):
        pass

    def encode_text(self, prompts, **kwargs):
        pass

    def get_variance(self, timestep, prev_timestep):
        pass

    def get_alpha_prod_t_prev(self, prev_timestep):
        pass

    def get_noise_shape(self, x0, num_steps):
        return (num_steps, self.model.unet.config.in_channels, x0.shape[-2], x0.shape[-1])

    def sample_xts_from_x0(self, x0, num_inference_steps = 50):
        alpha_bar = self.model.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        timesteps = self.model.scheduler.timesteps.to(self.device)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(self.get_noise_shape(x0, num_inference_steps + 1)).to(x0.device)
        xts[0] = x0

        for t in reversed(timesteps):
            idx = num_inference_steps - t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]

        return xts

    def get_zs_from_xts(self, xt, xtm1, noise_pred, t, eta = 0, numerical_fix = True, **kwargs):
        alpha_bar = self.model.scheduler.alphas_cumprod

        if self.model.scheduler.config.prediction_type == 'epsilon': pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
        elif self.model.scheduler.config.prediction_type == 'v_prediction': pred_original_sample = (alpha_bar[t] ** 0.5) * xt - ((1 - alpha_bar[t]) ** 0.5) * noise_pred

        prev_timestep = t - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        variance = self.get_variance(t, prev_timestep)

        if self.model.scheduler.config.prediction_type == 'epsilon': radom_noise_pred = noise_pred
        elif self.model.scheduler.config.prediction_type == 'v_prediction': radom_noise_pred = (alpha_bar[t] ** 0.5) * noise_pred + ((1 - alpha_bar[t]) ** 0.5) * xt

        mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + ((1 - alpha_prod_t_prev - eta * variance) ** (0.5) * radom_noise_pred)
        z = (xtm1 - mu_xt) / (eta * variance ** 0.5)

        if numerical_fix: xtm1 = mu_xt + (eta * variance ** 0.5)*z
        return z, xtm1, None

    def reverse_step_with_custom_noise(self, model_output, timestep, sample, variance_noise = None, eta = 0, **kwargs):
        prev_timestep = timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t

        if self.model.scheduler.config.prediction_type == 'epsilon': pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.model.scheduler.config.prediction_type == 'v_prediction': pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output

        variance = self.get_variance(timestep, prev_timestep)

        if self.model.scheduler.config.prediction_type == 'epsilon': model_output_direction = model_output
        elif self.model.scheduler.config.prediction_type == 'v_prediction': model_output_direction = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + ((1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction)

        if eta > 0:
            if variance_noise is None: variance_noise = torch.randn(model_output.shape, device=self.device)
            prev_sample = prev_sample + (eta * variance ** (0.5) * variance_noise)

        return prev_sample

    def unet_forward(self, sample, timestep, encoder_hidden_states, class_labels = None, timestep_cond = None, attention_mask = None, cross_attention_kwargs = None, added_cond_kwargs = None, down_block_additional_residuals = None, mid_block_additional_residual = None, encoder_attention_mask = None, replace_h_space = None, replace_skip_conns = None, return_dict = True, zero_out_resconns = None):
        pass

class STFT(torch.nn.Module):
    def __init__(self, fft_size, hop_size, window_size, window_type="hann"):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size
        self.window_type = window_type
        
        scale = fft_size / hop_size
        fourier_basis = np.fft.fft(np.eye(fft_size))

        cutoff = fft_size // 2 + 1
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        
        self.forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        self.inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        
        if window_type:
            assert fft_size >= window_size

            fft_window = torch.from_numpy(pad_center(get_window(window_type, window_size, fftbins=True), size=fft_size)).float()
            self.forward_basis *= fft_window
            self.inverse_basis *= fft_window
        
        if not hasattr(self, "forward_basis"): self.register_buffer("forward_basis", self.forward_basis)
        if not hasattr(self, "inverse_basis"): self.register_buffer("inverse_basis", self.inverse_basis)
    
    def transform(self, signal):
        batch_size, num_samples = signal.shape
        transformed_signal = F.conv1d(F.pad(signal.view(batch_size, 1, num_samples).unsqueeze(1), (self.fft_size // 2, self.fft_size // 2, 0, 0), mode="reflect").squeeze(1), self.forward_basis, stride=self.hop_size, padding=0).cpu()
        
        cutoff = self.fft_size // 2 + 1
        real_part, imag_part = transformed_signal[:, :cutoff, :], transformed_signal[:, cutoff:, :]

        return torch.sqrt(real_part ** 2 + imag_part ** 2), torch.atan2(imag_part, real_part)

class MelSpectrogramProcessor(torch.nn.Module):
    def __init__(self, fft_size, hop_size, window_size, num_mel_bins, sample_rate, fmin, fmax):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.stft_processor = STFT(fft_size, hop_size, window_size)
        self.register_buffer("mel_filter", torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mel_bins, fmin=fmin, fmax=fmax)).float())
    
    def compute_mel_spectrogram(self, waveform, normalization_fn=torch.log):
        assert torch.min(waveform) >= -1
        assert torch.max(waveform) <= 1
        
        magnitudes, _ = self.stft_processor.transform(waveform)
        return normalization_fn(torch.clamp(torch.matmul(self.mel_filter, magnitudes), min=1e-5))

class AudioLDM2(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AudioLDM2Pipeline.from_pretrained(self.model_id, local_files_only=True, torch_dtype=torch.float16 if config.is_half else torch.float32).to(self.device)

    def load_scheduler(self):
        self.model.scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

    def get_melspectrogram(self):
        return MelSpectrogramProcessor(fft_size=1024, hop_size=160, window_size=1024, num_mel_bins=64, sample_rate=16000, fmin=0, fmax=8000)

    def vae_encode(self, x):
        if x.shape[2] % 4: x = torch.nn.functional.pad(x, (0, 0, 4 - (x.shape[2] % 4), 0))
        return (self.model.vae.encode(x).latent_dist.mode() * self.model.vae.config.scaling_factor).float()

    def vae_decode(self, x):
        return self.model.vae.decode(1 / self.model.vae.config.scaling_factor * x).sample

    def decode_to_mel(self, x):
        tmp = self.model.mel_spectrogram_to_waveform(x[:, 0].detach().float()).detach()

        if len(tmp.shape) == 1: tmp = tmp.unsqueeze(0)
        return tmp

    def encode_text(self, prompts, negative = False, save_compute = False, cond_length = 0, **kwargs):
        tokenizers, text_encoders = [self.model.tokenizer, self.model.tokenizer_2], [self.model.text_encoder, self.model.text_encoder_2]
        prompt_embeds_list, attention_mask_list = [], []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(prompts, padding="max_length" if (save_compute and negative) or isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True, max_length=tokenizer.model_max_length if (not save_compute) or ((not negative) or isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer))) else cond_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids

            attention_mask = text_inputs.attention_mask
            untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids): tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])

            text_input_ids = text_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                if text_encoder.config.model_type == "clap":
                    prompt_embeds = text_encoder.get_text_features(text_input_ids, attention_mask=attention_mask)
                    prompt_embeds = prompt_embeds[:, None, :]
                    attention_mask = attention_mask.new_ones((len(prompts), 1))
                else: prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0]

            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)

        projection_output = self.model.projection_model(hidden_states=prompt_embeds_list[0], hidden_states_1=prompt_embeds_list[1], attention_mask=attention_mask_list[0], attention_mask_1=attention_mask_list[1])
        generated_prompt_embeds = self.model.generate_language_model(projection_output.hidden_states, attention_mask=projection_output.attention_mask, max_new_tokens=None)
        prompt_embeds = prompt_embeds.to(dtype=self.model.text_encoder_2.dtype, device=self.device)
        return generated_prompt_embeds.to(dtype=self.model.language_model.dtype, device=self.device), prompt_embeds, (attention_mask.to(device=self.device) if attention_mask is not None else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device))

    def get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.get_alpha_prod_t_prev(prev_timestep)
        return ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)

    def get_alpha_prod_t_prev(self, prev_timestep):
        return self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.model.scheduler.final_alpha_cumprod

    def unet_forward(self, sample, timestep, encoder_hidden_states, timestep_cond = None, class_labels = None, attention_mask = None, encoder_attention_mask = None, return_dict = True, cross_attention_kwargs = None, mid_block_additional_residual = None, replace_h_space = None, replace_skip_conns = None, zero_out_resconns = None):
        encoder_hidden_states_1 = class_labels
        class_labels = None
        encoder_attention_mask_1 = encoder_attention_mask
        encoder_attention_mask = None
        default_overall_up_factor = 2 ** self.model.unet.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]): forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if encoder_attention_mask_1 is not None:
            encoder_attention_mask_1 = (1 - encoder_attention_mask_1.to(sample.dtype)) * -10000.0
            encoder_attention_mask_1 = encoder_attention_mask_1.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"

            if isinstance(timestep, float): dtype = torch.float16 if is_mps else torch.float32
            else: dtype = torch.int16 if is_mps else torch.int32

            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0: timesteps = timesteps[None].to(sample.device)

        emb = self.model.unet.time_embedding(self.model.unet.time_proj(timesteps.expand(sample.shape[0])).to(dtype=sample.dtype), timestep_cond)
        aug_emb = None

        if self.model.unet.class_embedding is not None:
            if class_labels is None: raise ValueError

            if self.model.unet.config.class_embed_type == "timestep": class_labels = self.model.unet.time_proj(class_labels).to(dtype=sample.dtype)
            class_emb = self.model.unet.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.model.unet.config.class_embeddings_concat: emb = torch.cat([emb, class_emb], dim=-1)
            else: emb = emb + class_emb

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.model.unet.time_embed_act is not None: emb = self.model.unet.time_embed_act(emb)

        sample = self.model.unet.conv_in(sample)
        down_block_res_samples = (sample,)

        for downsample_block in self.model.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention: sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs, encoder_attention_mask=encoder_attention_mask, encoder_hidden_states_1=encoder_hidden_states_1, encoder_attention_mask_1=encoder_attention_mask_1)
            else: sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if self.model.unet.mid_block is not None: sample = self.model.unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs, encoder_attention_mask=encoder_attention_mask, encoder_hidden_states_1=encoder_hidden_states_1, encoder_attention_mask_1=encoder_attention_mask_1)

        if replace_h_space is None: h_space = sample.clone()
        else:
            h_space = replace_h_space
            sample = replace_h_space.clone()

        if mid_block_additional_residual is not None: sample = sample + mid_block_additional_residual
        extracted_res_conns = {}

        for i, upsample_block in enumerate(self.model.unet.up_blocks):
            is_final_block = i == len(self.model.unet.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if replace_skip_conns is not None and replace_skip_conns.get(i): res_samples = replace_skip_conns.get(i)

            if zero_out_resconns is not None:
                if (type(zero_out_resconns) is int and i >= (zero_out_resconns - 1)) or type(zero_out_resconns) is list and i in zero_out_resconns: res_samples = [torch.zeros_like(x) for x in res_samples]

            extracted_res_conns[i] = res_samples
            if not is_final_block and forward_upsample_size: upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention: sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, upsample_size=upsample_size, attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask, encoder_hidden_states_1=encoder_hidden_states_1, encoder_attention_mask_1=encoder_attention_mask_1)
            else: sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)

        if self.model.unet.conv_norm_out: sample = self.model.unet.conv_act(self.model.unet.conv_norm_out(sample))
        sample = self.model.unet.conv_out(sample)

        if not return_dict: return (sample,)
        return UNet2DConditionOutput(sample=sample), h_space, extracted_res_conns

def load_model(model, device):
    check_audioldm2(model)

    ldm_stable = AudioLDM2(model_id=os.path.join("assets", "models", "audioldm2", model), device=device, double_precision=False)
    ldm_stable.load_scheduler()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

    return ldm_stable