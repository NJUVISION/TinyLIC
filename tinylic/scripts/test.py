


import os
import sys
import time
import json
import math
import shutil
import argparse
import importlib
from typing import List
from PIL import Image
from collections import defaultdict

import lpips
import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_msssim import ms_ssim

from torchmetrics.image import (
    FrechetInceptionDistance,
)
from eval._update_patch_fid import update_patch_fid

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def image_models(model):
    model_map = {
        'tinylic_10k': 'models.tinylic_10k.FastNIC',
        'tinylic_20k': 'models.tinylic_20k.FastNIC',
        'tinylic_50k': 'models.tinylic_50k.FastNIC',
    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, qp, fid_metric, lpips_metrics, img_name, save, arch):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    
    lpips_alex, lpips_vgg = lpips_metrics(x * 2 - 1, out_dec["x_hat"] * 2 - 1)
    update_patch_fid(x, out_dec["x_hat"], fid_metric=fid_metric)    
    
    
    

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    if save:
        img = transforms.ToPILImage(mode='RGB')(out_dec["x_hat"].squeeze(0))
        img.save(os.path.join(f'./rec/{arch}/qp{qp}', f"{img_name}.png"))
    
    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "lpips_alex": lpips_alex.item(),
        "lpips_vgg": lpips_vgg.item(),
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, qp, fid_metric, img_name, save, arch):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": psnr(x, out_net["x_hat"].clamp_(0, 1)),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,
        "decoding_time": elapsed_time / 2.0,
    }




class LPIPSMetrics(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.alex = lpips.LPIPS(net="alex").to(device).eval()
        self.vgg = lpips.LPIPS(net="vgg").to(device).eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, inputs, reconstructions):
        alex = self.alex(inputs, reconstructions).mean()
        vgg = self.vgg(inputs, reconstructions).mean()
        return alex, vgg


def eval_model(model, qp, filepaths, entropy_estimation=False, half=False, save=False, arch=None):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    fid_metric = FrechetInceptionDistance().to(device)
    lpips_metrics = LPIPSMetrics(device)

    if save:
        rec_dir = f"./rec/{arch}/qp{qp}"
        if os.path.exists(rec_dir):
            shutil.rmtree(rec_dir)
        os.makedirs(rec_dir, exist_ok=True)

    for f in filepaths:
        x = read_image(f).to(device)
        img_name = os.path.splitext(os.path.basename(f))[0]

        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, qp, fid_metric, lpips_metrics, img_name=img_name, save=save, arch=arch)
        else:
            rv = inference_entropy_estimation(model, x, qp, fid_metric, img_name=img_name, save=save, arch=arch)

        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    metrics['fid'] = float(fid_metric.compute())
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument("--dataset", type=str, help="dataset path")
    parent_parser.add_argument("--path", type=str, help="ckpt path")
    parent_parser.add_argument(
        "-a", "--architecture", type=str, help="model architecture", required=True
    )
    parent_parser.add_argument(
        "--half", action="store_true", help="convert model to half floating point (fp16)"
    )
    parent_parser.add_argument(
        "--entropy-estimation", action="store_true", help="use evaluated entropy estimation (no entropy coding)"
    )
    parent_parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose mode"
    )
    parent_parser.add_argument(
        "--save", action="store_true", help="save mode"
    )
    parent_parser.add_argument(
        "--fuse-repconv", action="store_true", help="fuse RepConv branches before evaluation"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.",
        add_help=True,
        parents=[parent_parser]
    )
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)
    

    results = defaultdict(list)
    model = image_models(args.architecture).eval()

    qp = os.path.basename(os.path.dirname(args.path))
    
    checkpoint = torch.load(args.path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema2" in checkpoint:
        state_dict = checkpoint["ema2"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("epoch=", checkpoint["epoch"])
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda")
    model.update(force=True)
    if args.fuse_repconv:
        model.fuse_repconv()

    metrics = eval_model(model, qp, filepaths, args.entropy_estimation, args.half, args.save, args.architecture)
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": args.architecture,
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
