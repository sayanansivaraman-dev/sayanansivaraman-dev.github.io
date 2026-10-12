"""Regenerate the figures on projects/gwhd_2021.md.

    cd <this repo>
    uv run --project ../agriculture_perception/projects/p3_veg_det python _tools/gwhd_figures.py

Needs a logged-in W&B CLI (`wandb login`) and the p3 project's venv for `wandb`, matplotlib and
PIL. Writes into assets/gwhd_2021/:

    val_ap_curves.png        val AP per epoch, pulled live from W&B run history
    test_ap_per_domain.png   test AP per held-out domain, from gwhd_2021_test_eval.json
    detections_val.jpg       GT vs each arm on the four val frames W&B logged at each best step

split_examples.jpg and detections_test.jpg need the GWHD images, which only live on the RunPod
volume; gwhd_pod_examples.py renders those on a pod (a CPU pod is enough).

Val numbers come from W&B rather than being typed in, so a figure cannot drift from its run. Test
numbers only exist in the eval_test.json files on the RunPod volume (the eval-only path does not
log to W&B), so a verbatim copy is committed beside this script.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

SITE = Path(__file__).resolve().parent.parent
OUT = SITE / "assets" / "gwhd_2021"
PROJECT = "sayanan-sivaraman-personal/agriperc-p3-det"

#: The three Experiment 1 arms (final pipeline, agriculture_perception 2a2b3b8).
ARMS = {
    "ConvNeXt-T s4": "v98sommh",
    "ConvNeXt-T s16": "u80ze644",
    "ViT-S+/16": "pqsdcdh6",
}
#: The same s4 arm before the pixel-Gaussian / fp32-head / peak-window fixes.
S4_BEFORE = "4bfwhdpk"
TEST_KEYS = {"ConvNeXt-T s4": "convnext_s4", "ConvNeXt-T s16": "convnext_s16", "ViT-S+/16": "vit"}

# Site tokens (assets/css/style.css, light) and the dataviz skill's categorical slots 1-3, which
# validate all-pairs for CVD in light mode. Aqua is under 3:1 on white, so every series is also
# direct-labelled or legended and the page carries the numbers in tables.
INK, INK_SOFT, INK_FAINT, RULE, BG = "#1a1c1e", "#55595e", "#8b9096", "#e2e5e8", "#ffffff"
SERIES = {"ConvNeXt-T s4": "#2a78d6", "ConvNeXt-T s16": "#eb6834", "ViT-S+/16": "#1baf7a"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def val_history(api: wandb.Api, run_id: str) -> list[tuple[int, float]]:
    run = api.run(f"{PROJECT}/{run_id}")
    rows = run.scan_history(keys=["_step", "val/ap"])
    return sorted((r["_step"], r["val/ap"]) for r in rows if r.get("val/ap") is not None)


def val_curves(api: wandb.Api) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)

    before = val_history(api, S4_BEFORE)
    epochs = list(range(1, len(before) + 1))
    ax.plot(epochs, [ap for _, ap in before], color=INK_FAINT, linewidth=2,
            linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("s4 before the fixes", (epochs[-1], before[-1][1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK_FAINT, fontsize=9)

    # Hand-placed so the three end labels never collide; revisit if the curves change.
    label_dy = {"ConvNeXt-T s4": -9, "ConvNeXt-T s16": 5, "ViT-S+/16": -3}
    for name, run_id in ARMS.items():
        ys = [ap for _, ap in val_history(api, run_id)]
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, color=SERIES[name], linewidth=2, zorder=3)
        ax.scatter(xs, ys, s=22, color=SERIES[name], edgecolor=BG, linewidth=1.5, zorder=4)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, label_dy[name]),
                    textcoords="offset points", va="center", color=INK, fontsize=9)

    ax.set_xlim(0.7, len(epochs) + 2.4)
    ax.set_xticks(epochs)
    ax.set_xlabel("epoch (915 steps each)")
    ax.set_ylabel("val AP @[.5:.95]")
    ax.set_ylim(0.12, 0.30)
    ax.set_title("Validation AP per epoch, conv head", loc="left", color=INK, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "val_ap_curves.png")
    plt.close(fig)


def test_per_domain() -> None:
    data = json.loads((SITE / "_tools" / "gwhd_2021_test_eval.json").read_text())["arms"]
    names = list(SERIES)
    per = {n: data[TEST_KEYS[n]]["ap_per_domain"] for n in names}
    domains = sorted(per[names[0]], key=lambda d: sum(per[n][d] for n in names) / len(names))

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=200)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    # Each arm on its own sub-row, so exact ties (all three 0.180 on Ukyoto_1) stay visible.
    offsets = (-0.22, 0.0, 0.22)
    for row, dom in enumerate(domains):
        vals = [per[n][dom] for n in names]
        ax.plot([min(vals), max(vals)], [row, row], color=RULE, linewidth=2, zorder=1)
        for i, name in enumerate(names):
            ax.scatter(vals[i], row + offsets[i], s=40, color=SERIES[name], edgecolor=BG,
                       linewidth=1.2, zorder=3 + i)
    ax.set_yticks(range(len(domains)))
    ax.set_ylim(-0.6, len(domains) - 0.4)
    ax.set_yticklabels(domains, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("test AP @[.5:.95]")
    ax.set_xlim(0.0, 0.30)
    ax.set_title("Test AP per held-out domain (sorted by mean)", loc="left", color=INK,
                 fontsize=11, pad=24)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=SERIES[n],
                          markeredgecolor=BG, label=n) for n in names]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=9, handletextpad=0.3, columnspacing=1.4, borderaxespad=0.2,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "test_ap_per_domain.png")
    plt.close(fig)


def best_panels(api: wandb.Api, run_id: str, scratch: Path) -> list[tuple[str, Image.Image]]:
    """(caption, GT|pred image) for each frame logged at the run's best-val-AP step."""
    run = api.run(f"{PROJECT}/{run_id}")
    best_step = max(val_history(api, run_id), key=lambda sa: sa[1])[0]
    row = next(r for r in run.scan_history(keys=["_step", "val/detections"])
               if r["_step"] == best_step)["val/detections"]
    panels = []
    for name, caption in zip(row["filenames"], row["captions"], strict=True):
        handle = run.file(name).download(root=str(scratch / run_id), replace=True)
        panels.append((caption, Image.open(io.BytesIO(Path(handle.name).read_bytes())).copy()))
        handle.close()
    return panels


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def compose_grid(rows: list[list[tuple[str, Image.Image]]], out: Path, *, tile: int = 420) -> None:
    """Labelled image grid on the page background. Shared with gwhd_pod_examples.py."""
    pad, head = 12, 30
    cols = max(len(r) for r in rows)
    width = cols * tile + (cols + 1) * pad
    height = len(rows) * (tile + head) + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    for r, cells in enumerate(rows):
        y = pad + r * (tile + head + pad)
        for c, (label, img) in enumerate(cells):
            x = pad + c * (tile + pad)
            draw.text((x, y + 5), label, fill=INK, font=font)
            canvas.paste(img.resize((tile, tile), Image.LANCZOS), (x, y + head))
    canvas.save(out, quality=80, optimize=True, progressive=True)


def panel_row(arm_panels: dict[str, tuple[str, Image.Image]]) -> list[tuple[str, Image.Image]]:
    """One grid row from each arm's GT|pred panel of the same frame: GT once, then predictions.

    Captions follow train.detection_panels: "<domain> | GT <n> | pred <m> @>0.3".
    """
    caption, first = next(iter(arm_panels.values()))
    domain, gt = (part.strip() for part in caption.split("|")[:2])
    half = first.width // 2
    cells = [(f"{domain}: ground truth, {gt.split()[1]} heads", first.crop((0, 0, half, first.height)))]
    for name, (cap, img) in arm_panels.items():
        count = cap.split("|")[2].split()[1]
        cells.append((f"{name}: {count} boxes", img.crop((half, 0, img.width, img.height))))
    return cells


def detection_grid(api: wandb.Api) -> None:
    """Rows: the logged val frames. Columns: ground truth, then each arm's predictions > 0.3."""
    with tempfile.TemporaryDirectory() as scratch:
        arms = {name: best_panels(api, run_id, Path(scratch)) for name, run_id in ARMS.items()}
    frames = len(next(iter(arms.values())))
    rows = [panel_row({name: panels[r] for name, panels in arms.items()}) for r in range(frames)]
    compose_grid(rows, OUT / "detections_val.jpg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    val_curves(api)
    test_per_domain()
    detection_grid(api)
    for path in sorted(OUT.glob("*.*")):
        print(f"wrote {path.relative_to(SITE)}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
