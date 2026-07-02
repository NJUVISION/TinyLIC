import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_TITLES = {
    "kodak": "Kodak",
    "clic2020": "CLIC2020",
    "jpegai_test_chunk": "JPEG AI Test Chunk",
    "tecnick": "Tecnick",
}

MODEL_LABELS = ("tinylic_10k", "tinylic_20k", "tinylic_50k")

METHOD_STYLES = {
    "VTM": {"color": "#111827", "linestyle": "-", "marker": "s", "linewidth": 1.8, "alpha": 0.82},
    "BPG": {"color": "#8a8f98", "linestyle": "-", "marker": "x", "linewidth": 1.5, "alpha": 0.68},
    "tinylic_10k": {"color": "#2563eb", "linestyle": "-", "marker": "o", "linewidth": 2.25, "alpha": 1.0},
    "tinylic_20k": {"color": "#dc2626", "linestyle": "-", "marker": "^", "linewidth": 2.25, "alpha": 1.0},
    "tinylic_50k": {"color": "#16a34a", "linestyle": "-", "marker": "D", "linewidth": 2.25, "alpha": 1.0},
}


def bd_rate(anchor_psnr, anchor_bpp, test_psnr, test_bpp):
    anchor_psnr = np.asarray(anchor_psnr, dtype=np.float64)
    anchor_bpp = np.asarray(anchor_bpp, dtype=np.float64)
    test_psnr = np.asarray(test_psnr, dtype=np.float64)
    test_bpp = np.asarray(test_bpp, dtype=np.float64)

    order_anchor = np.argsort(anchor_psnr)
    order_test = np.argsort(test_psnr)
    anchor_psnr = anchor_psnr[order_anchor]
    anchor_bpp = anchor_bpp[order_anchor]
    test_psnr = test_psnr[order_test]
    test_bpp = test_bpp[order_test]

    psnr_min = max(anchor_psnr.min(), test_psnr.min())
    psnr_max = min(anchor_psnr.max(), test_psnr.max())
    if psnr_max <= psnr_min:
        return None

    degree = min(3, len(anchor_psnr) - 1, len(test_psnr) - 1)
    anchor_poly = np.polyfit(anchor_psnr, np.log(anchor_bpp), degree)
    test_poly = np.polyfit(test_psnr, np.log(test_bpp), degree)
    anchor_int = np.poly1d(np.polyint(anchor_poly))
    test_int = np.poly1d(np.polyint(test_poly))
    avg = (
        test_int(psnr_max)
        - test_int(psnr_min)
        - anchor_int(psnr_max)
        + anchor_int(psnr_min)
    ) / (psnr_max - psnr_min)
    return float((np.exp(avg) - 1.0) * 100.0)


def load_baseline_curves(baselines_path):
    with baselines_path.open() as f:
        return json.load(f)


def load_fastnic_curves(results_path, dataset):
    with results_path.open() as f:
        results = json.load(f)
    curves = {}
    for method, points in results["datasets"][dataset]["methods"].items():
        curves[method] = {
            "bpp": [item["bpp"] for item in points],
            "psnr": [item["psnr"] for item in points],
        }
    return curves


def axis_limits_from_models(curves):
    bpp = np.concatenate([np.asarray(curves[label]["bpp"], dtype=np.float64) for label in MODEL_LABELS])
    psnr = np.concatenate([np.asarray(curves[label]["psnr"], dtype=np.float64) for label in MODEL_LABELS])

    x_min, x_max = float(bpp.min()), float(bpp.max())
    y_min, y_max = float(psnr.min()), float(psnr.max())
    x_pad = max((x_max - x_min) * 0.10, 0.01)
    y_pad = max((y_max - y_min) * 0.12, 0.25)

    return (max(0.0, x_min - x_pad), x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_dataset(dataset, curves, out_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.set_facecolor("#fbfbfc")
    ax.grid(True, which="major", color="#d7dce2", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#eceff3", linewidth=0.5, alpha=0.8)
    ax.minorticks_on()
    ax.set_xlabel("Bits per pixel (BPP)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(DATASET_TITLES[dataset], pad=10, fontweight="bold")

    xlim, ylim = axis_limits_from_models(curves)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    for label, values in curves.items():
        style = METHOD_STYLES.get(label, {})
        is_model = label in MODEL_LABELS
        ax.plot(
            values["bpp"],
            values["psnr"],
            label=label,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=style.get("linewidth", 1.6),
            markersize=6 if is_model else 5,
            markerfacecolor="white" if is_model else style.get("color"),
            markeredgewidth=1.4,
            alpha=style.get("alpha", 1.0),
            zorder=3 if is_model else 2,
        )

    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.9)

    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#d1d5db",
        facecolor="white",
        borderpad=0.65,
        handlelength=2.4,
    )
    png_path = out_dir / f"{dataset}_rd.png"
    pdf_path = out_dir / f"{dataset}_rd.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("utils/draw/rd_results.json"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("utils/draw/rd_baselines.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("utils/draw"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = load_baseline_curves(args.baselines)
    bd_summary = {}
    for dataset in baseline_data:
        curves = baseline_data[dataset].copy()
        curves.update(load_fastnic_curves(args.results, dataset))

        if "VTM" not in curves:
            raise ValueError(f"{dataset} has no VTM anchor curve")

        bd_summary[dataset] = {}
        for label, values in curves.items():
            if label == "VTM":
                bd_summary[dataset][label] = 0.0
                continue
            bd_summary[dataset][label] = bd_rate(
                curves["VTM"]["psnr"],
                curves["VTM"]["bpp"],
                values["psnr"],
                values["bpp"],
            )

        plot_dataset(dataset, curves, out_dir)

    bd_path = out_dir / "bd_rate_vtm_anchor.json"
    with bd_path.open("w") as f:
        json.dump(bd_summary, f, indent=2)

    print(json.dumps(bd_summary, indent=2))


if __name__ == "__main__":
    main()
