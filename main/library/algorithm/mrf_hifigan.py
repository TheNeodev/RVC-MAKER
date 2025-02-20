import torch

import numpy as np
import torch.nn.functional as F

from torch.nn.utils import remove_weight_norm
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1

class MRFLayer(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size * dilation - dilation) // 2, dilation=dilation))
        self.conv2 = weight_norm(torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, dilation=1))

    def forward(self, x):
        return x + self.conv2(F.leaky_relu(self.conv1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)

class MRFBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()

class SineGenerator(torch.nn.Module):
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGenerator, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)

        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0

        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        return torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]

            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)

            sine_waves = sine_waves * uv + ((uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sine_waves))

        return sine_waves

class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        
        self.l_sin_gen = SineGenerator(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x).to(dtype=self.l_linear.weight.dtype)))

class HiFiGANMRFGenerator(torch.nn.Module):
    def __init__(self, in_channel, upsample_initial_channel, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilations, gin_channels, sample_rate, harmonic_num, checkpointing=False):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.upp = int(np.prod(upsample_rates))
        self.f0_upsample = torch.nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        self.conv_pre = weight_norm(torch.nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.checkpointing = checkpointing

        self.upsamples = torch.nn.ModuleList()
        self.upsampler = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        stride_f0s = [upsample_rates[1] * upsample_rates[2] * upsample_rates[3], upsample_rates[2] * upsample_rates[3], upsample_rates[3], 1]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if self.upp == 441:
                self.upsampler.append(torch.nn.Upsample(scale_factor=u, mode="linear"))
                self.upsamples.append(weight_norm(torch.nn.Conv1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), kernel_size=1)))
                self.noise_convs.append(torch.nn.Conv1d(in_channels=1, out_channels=upsample_initial_channel // (2 ** (i + 1)), kernel_size = 1))
            else:
                self.upsampler.append(torch.nn.Identity())
                self.upsamples.append(weight_norm(torch.nn.ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), kernel_size=k, stride=u, padding=(k - u) // 2)))
                self.noise_convs.append(torch.nn.Conv1d(1, upsample_initial_channel // (2 ** (i + 1)), kernel_size=stride_f0s[i] * 2 if stride_f0s[i] > 1 else 1, stride=stride_f0s[i], padding=stride_f0s[i] // 2))

        self.mrfs = torch.nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(torch.nn.ModuleList([MRFBlock(channel, kernel_size=k, dilations=d) for k, d in zip(resblock_kernel_sizes, resblock_dilations)]))

        self.conv_post = weight_norm(torch.nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, g = None):
        har_source = self.m_source(self.f0_upsample(f0[:, None, :]).transpose(-1, -2)).transpose(-1, -2)
        x = self.conv_pre(x)
        if g is not None: x += self.cond(g)

        for ups, upr, mrf, noise_conv in zip(self.upsamples, self.upsampler, self.mrfs, self.noise_convs):
            x = F.leaky_relu(x, LRELU_SLOPE)

            if self.training and self.checkpointing:
                if self.upp == 441: x = upr(x)
                x = checkpoint(ups, x, use_reentrant=False)
            else:
                if self.upp == 441: x = upr(x)
                x = ups(x)

            h = noise_conv(har_source)
            if self.upp == 441: h = torch.nn.functional.interpolate(h, size=x.shape[-1], mode="linear")
            x += h

            def mrf_sum(x, layers):
                return sum(layer(x) for layer in layers) / self.num_kernels
            
            x = checkpoint(mrf_sum, x, mrf, use_reentrant=False) if self.training and self.checkpointing else mrf_sum(x, mrf)

        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)

        for up in self.upsamples:
            remove_weight_norm(up)

        for mrf in self.mrfs:
            mrf.remove_weight_norm()

        remove_weight_norm(self.conv_post)