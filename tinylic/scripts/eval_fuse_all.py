import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchvision import transforms


DATASETS = {
    "clic2020": "datasets/clic2020",
    "jpegai_test_chunk": "datasets/jpegai_test_chunk",
    "tecnick": "datasets/tecnick",
    "kodak": "../Kodak",
}

ARCHITECTURES = ("tinylic_10k", "tinylic_20k", "tinylic_50k")
RATE_POINTS = ("1", "2", "3", "4")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def image_model(name):
    module = importlib.import_module(f"models.{name}")
    return module.FastNIC()


def collect_images(root):
    return sorted(
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def read_image(path):
    return transforms.ToTensor()(Image.open(path).convert("RGB"))


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


@torch.no_grad()
def run_image(model, x):
    x = x.unsqueeze(0)
    height, width = x.size(2), x.size(3)
    pad = 64
    new_height = (height + pad - 1) // pad * pad
    new_width = (width + pad - 1) // pad * pad
    padding_left = (new_width - width) // 2
    padding_right = new_width - width - padding_left
    padding_top = (new_height - height) // 2
    padding_bottom = new_height - height - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    out_enc = model.compress(x_padded)

    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    x_hat = F.pad(
        out_dec["x_hat"],
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
    ).clamp_(0, 1)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    return {
        "psnr": psnr(x, x_hat),
        "ms-ssim": ms_ssim(x, x_hat, data_range=1.0).item(),
        "bpp": bpp,
    }


def load_checkpoint(model, ckpt):
    checkpoint = torch.load(ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema2" in checkpoint:
        state_dict = checkpoint["ema2"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


def run_eval(root, dataset_name, filepaths, architecture, rate_point, cuda_device):
    ckpt = root / "ckpts" / architecture / f"{rate_point}.pth.tar"
    print(f"[eval] {dataset_name} {architecture} rate={rate_point}", flush=True)
    torch.cuda.set_device(int(cuda_device))
    model = image_model(architecture).eval()
    load_checkpoint(model, ckpt)
    model = model.to("cuda")
    model.update(force=True)
    model.fuse_repconv()

    totals = {}
    for path in filepaths:
        x = read_image(path).to("cuda")
        metrics = run_image(model, x)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value

    metrics = {key: value / len(filepaths) for key, value in totals.items()}
    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("utils/draw/rd_results.json"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["OMP_NUM_THREADS"] = "1"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with out_path.open() as f:
            output = json.load(f)
        output.setdefault("datasets", {})
    else:
        output = {"datasets": {}}

    for dataset_name in args.datasets:
        rel_path = DATASETS[dataset_name]
        dataset_path = (root / rel_path).resolve()
        filepaths = collect_images(dataset_path)
        if not filepaths:
            raise ValueError(f"No images found for {dataset_name}: {dataset_path}")
        output["datasets"][dataset_name] = {
            "methods": {},
        }
        for architecture in ARCHITECTURES:
            output["datasets"][dataset_name]["methods"][architecture] = []
            for rate_point in RATE_POINTS:
                metrics = run_eval(
                    root,
                    dataset_name,
                    filepaths,
                    architecture,
                    rate_point,
                    args.cuda_device,
                )
                output["datasets"][dataset_name]["methods"][architecture].append(metrics)
                with out_path.open("w") as f:
                    json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
