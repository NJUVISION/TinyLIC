import argparse
import importlib
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_MAP = OrderedDict(
    [
        ("tinylic_10k", "models.tinylic_10k.FastNIC"),
        ("tinylic_20k", "models.tinylic_20k.FastNIC"),
        ("tinylic_50k", "models.tinylic_50k.FastNIC"),
    ]
)
DEFAULT_MODELS = list(MODEL_MAP)


class BackboneEncode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.compress_backbone(x)


class BackboneDecode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_hat, y_hat):
        return self.model.decompress_backbone(z_hat, y_hat)


def import_model(name):
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model {name}. Choices: {', '.join(MODEL_MAP)}")
    module_path, class_name = MODEL_MAP[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model = getattr(module, class_name)().eval()
    if hasattr(model, "update"):
        model.update()
    if hasattr(model, "fuse_repconv"):
        model.fuse_repconv()
    return model


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def profile_ops(module, inputs):
    analysis = FlopCountAnalysis(module, inputs)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    analysis.tracer_warnings("none")
    return analysis.total()


def kmac_per_pixel(ops, h, w):
    return ops / 1e3 / (h * w)


@torch.no_grad()
def analyze_model(model_name, input_size, device):
    model = import_model(model_name).to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    _, _, h, w = x.shape

    backbone_encode = BackboneEncode(model).to(device).eval()
    backbone_decode = BackboneDecode(model).to(device).eval()
    backbone_out = backbone_encode(x)
    z_hat, y_hat = backbone_out[0], backbone_out[1]

    enc_ops = profile_ops(backbone_encode, x)
    dec_ops = profile_ops(backbone_decode, (z_hat, y_hat))
    total_ops = enc_ops + dec_ops

    print("\n" + "=" * 88)
    print(f"Model: {model_name} | input: {input_size}x{input_size} | device: {device} | RepConv: fused")
    print(f"Case: backbone_with_entropy_likelihood")
    print(f"Params: {count_params(model) / 1e6:.4f} M")
    print(f"Latents: z_hat={tuple(z_hat.shape)}, y_hat={tuple(y_hat.shape)}")
    print("-" * 88)
    print(f"{'encode GMAC':>12s} {'decode GMAC':>12s} {'total GMAC':>12s} "
          f"{'encode KMAC/pix':>16s} {'decode KMAC/pix':>16s} {'total KMAC/pix':>16s}")
    print(
        f"{enc_ops / 1e9:12.6f} {dec_ops / 1e9:12.6f} {total_ops / 1e9:12.6f} "
        f"{kmac_per_pixel(enc_ops, h, w):16.6f} {kmac_per_pixel(dec_ops, h, w):16.6f} "
        f"{kmac_per_pixel(total_ops, h, w):16.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Profile backbone encode/decode complexity.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_MAP.keys()),
        help="Models to profile.",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Square input size.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU profiling.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    for model_name in args.models:
        analyze_model(model_name, args.input_size, device)


if __name__ == "__main__":
    main()
