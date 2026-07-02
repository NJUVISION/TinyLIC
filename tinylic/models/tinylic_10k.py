import math

import torch
import torch.nn as nn
from torch import Tensor

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .base import CompressionModel

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x

class WSiLUChunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        h = self.silu(x1) * x2
        return h

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        out = x * norm * self.weight.view(1, -1, 1, 1)
        return out

def quantize_ste(x: Tensor) -> Tensor:
    return (torch.round(x) - x).detach() + x

class RepConv(torch.nn.Module):
    def __init__(self, out_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch, bias=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 1, 1, 0, groups=out_ch, bias=True)

    def forward(self, x):
        return self.conv(x) + self.conv1(x)

    @torch.no_grad()
    def fuse(self):
        fused = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            kernel_size=3,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=True,
            padding_mode=self.conv.padding_mode,
        ).to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        conv1_weight = torch.nn.functional.pad(self.conv1.weight, [1, 1, 1, 1])
        fused.weight.copy_(self.conv.weight + conv1_weight)
        fused.bias.copy_(self.conv.bias + self.conv1.bias)
        return fused

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_div=1.5):
        super().__init__()
        self.dc = nn.Sequential(
            RepConv(out_ch),
        )
        self.ffn = nn.Sequential(
            RMSNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch*2, 1),
            WSiLUChunk(),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def forward(self, x):
        out = self.dc(x)
        out = self.ffn(out)
        out += x
        return out

class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = ResidualBlock(out_ch, out_ch, n_div=1)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out

class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, n_div=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )
        self.conv = ResidualBlock(out_ch, out_ch, n_div=n_div)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class ResidualBlockUpsample2(nn.Module):
    def __init__(self, in_ch, out_ch, n_div=1):
        super().__init__()
        self.conv = ResidualBlock(in_ch, in_ch, n_div=n_div)
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.up(out)
        return out

class FastNIC(CompressionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.g_a = nn.Sequential(
            nn.PixelUnshuffle(4),
            ResidualBlock(48, 48, n_div=1),
            nn.Conv2d(48, 140, kernel_size=2, stride=2),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            ResidualBlock(140, 140, n_div=1),
            nn.Conv2d(140, 256, kernel_size=2, stride=2),
            ResidualBlock(256, 256, n_div=1),
        )

        self.g_s = nn.Sequential(
            ResidualBlockUpsample(256, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlock(128, 128, n_div=1),
            ResidualBlockUpsample(128, 48, n_div=1),
            ResidualBlock(48, 48, n_div=1),
            nn.PixelShuffle(4),
        )

        self.h_a = nn.Sequential(
            ResidualBlockWithStride2(256, 192),
            ResidualBlock(192, 192, n_div=1),
            ResidualBlockWithStride2(192, 160),
        )

        self.h_s = nn.Sequential(
            ResidualBlockUpsample2(160, 192),
            ResidualBlock(192, 192, n_div=1),
            ResidualBlockUpsample2(192, 256),
        )

        self.num_slices = 8
        self.max_support_slices = 8

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(int(256 // self.num_slices) * self.max_support_slices, int(256 // self.num_slices) * self.max_support_slices, kernel_size=5, padding=2, stride=1, groups=int(256 // self.num_slices) * self.max_support_slices),
                nn.Conv2d(int(256 // self.num_slices) * self.max_support_slices, 128, 1),
                WSiLU(),
                nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1, groups=128),
                nn.Conv2d(128, int(128 // self.num_slices) * 2, 1),
                WSiLU(),
                nn.Conv2d(int(128 // self.num_slices) * 2, int(128 // self.num_slices) * 2, kernel_size=5, padding=2, stride=1, groups=int(128 // self.num_slices) * 2),
                nn.Conv2d(int(128 // self.num_slices) * 2, int(128 // self.num_slices) * 3, 1),
                WSiLU(),
                nn.Conv2d(int(128 // self.num_slices) * 3, int(256 // self.num_slices) * 2, 1),
            ) for i in range(self.num_slices)
        )

        self.LRP = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(int(256 // self.num_slices) * 2, int(256 // self.num_slices) * 2, kernel_size=5, padding=2, stride=1, groups=int(256 // self.num_slices) * 2),
                nn.Conv2d(int(256 // self.num_slices) * 2, int(256 // self.num_slices) * 3, 1),
                WSiLU(),
                nn.Conv2d(int(256 // self.num_slices) * 3, int(256 // self.num_slices) * 3, kernel_size=5, padding=2, stride=1, groups=int(256 // self.num_slices) * 3),
                nn.Conv2d(int(256 // self.num_slices) * 3, int(256 // self.num_slices) * 2, 1),
                WSiLU(),
                nn.Conv2d(int(256 // self.num_slices) * 2, int(256 // self.num_slices), 1),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(160)
        self.gaussian_conditional = GaussianConditional(None)

    @torch.no_grad()
    def fuse_repconv(self):
        def fuse_children(module):
            for name, child in list(module.named_children()):
                if isinstance(child, RepConv):
                    setattr(module, name, child.fuse())
                else:
                    fuse_children(child)

        fuse_children(self)
        return self

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        params = self.h_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        param_slices = list(params.chunk(self.num_slices, 1))
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            gaussian_params = self.cc_transforms[slice_index](torch.cat(param_slices[:self.max_support_slices], dim=1))
            scale, mu = gaussian_params.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            lrp = self.LRP[slice_index](torch.cat([y_hat_slice, param_slices[slice_index]], dim=1))
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            param_slices[slice_index] = y_hat_slice

        y_hat = torch.cat(param_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def build_indexes(self, scales: torch.Tensor) -> torch.Tensor:
        clamp_scales = torch.clamp(scales, 0.11, 256)
        v1 = math.log(0.11)
        v2 = 63 / (math.log(256) - v1)
        indexes = torch.ceil((torch.log(clamp_scales) - v1) * v2).int()
        return indexes

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        param_slices = list(params.chunk(self.num_slices, 1))

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            gaussian_params = self.cc_transforms[slice_index](torch.cat(param_slices[:self.max_support_slices], dim=1))
            scale, mu = gaussian_params.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp = self.LRP[slice_index](torch.cat([y_hat_slice, param_slices[slice_index]], dim=1))
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            param_slices[slice_index] = y_hat_slice

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        params = self.h_s(z_hat)

        y_string = strings[0][0]
        param_slices = list(params.chunk(self.num_slices, 1))

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            gaussian_params = self.cc_transforms[slice_index](torch.cat(param_slices[:self.max_support_slices], dim=1))
            scale, mu = gaussian_params.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, z_hat.shape[2]*4, z_hat.shape[3]*4)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp = self.LRP[slice_index](torch.cat([y_hat_slice, param_slices[slice_index]], dim=1))
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            param_slices[slice_index] = y_hat_slice

        y_hat = torch.cat(param_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def compress_backbone(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        param_slices = list(params.chunk(self.num_slices, 1))

        y_slices = y.chunk(self.num_slices, 1)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            gaussian_params = self.cc_transforms[slice_index](torch.cat(param_slices[:self.max_support_slices], dim=1))
            scale, mu = gaussian_params.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp = self.LRP[slice_index](torch.cat([y_hat_slice, param_slices[slice_index]], dim=1))
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            param_slices[slice_index] = y_hat_slice
        y_hat = torch.cat(param_slices, dim=1)

        return z_hat, y_hat, z_likelihoods, mu, scale

    def decompress_backbone(self, z_hat, y_hat):

        params = self.h_s(z_hat)
        param_slices = list(params.chunk(self.num_slices, 1))

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        y_slices = y_hat.chunk(self.num_slices, 1)
        for slice_index, y_hat_slice in enumerate(y_slices):
            gaussian_params = self.cc_transforms[slice_index](torch.cat(param_slices[:self.max_support_slices], dim=1))
            scale, mu = gaussian_params.chunk(2, 1)

            index = self.gaussian_conditional.build_indexes(scale)
            _, y_slice_likelihood = self.gaussian_conditional(y_hat_slice, scale, mu)

            lrp = self.LRP[slice_index](torch.cat([y_hat_slice, param_slices[slice_index]], dim=1))
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            param_slices[slice_index] = y_hat_slice

        y_hat = torch.cat(param_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat, mu, scale, index, y_slice_likelihood
