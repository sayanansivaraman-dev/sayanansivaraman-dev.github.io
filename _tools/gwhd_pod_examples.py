"""Figures that need the GWHD images themselves, which only live on the RunPod network volume.

Run ON A POD with the volume at /workspace, from the p3 project's venv, with this repo's
_tools/ copied alongside (it imports the shared grid helpers from gwhd_figures.py):

    python gwhd_pod_examples.py <out_dir>

Writes into <out_dir>, for copying back to assets/gwhd_2021/:

    split_examples.jpg   one image per split (train / val / test) with ground-truth boxes
    detections_test.jpg  GT vs each arm's best checkpoint on one image from each of three test
                         domains: the ViT's best (UQ_11), its worst (KSU_4), and the easiest (NAU_3)

CPU is enough: three heads over three 1024 px images. The trunk runs in fp32 here rather than the
bf16 it trained under, so scores near the 0.3 drawing threshold can differ slightly from a GPU run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from gwhd_figures import ARMS, OUT as _UNUSED, compose_grid, panel_row  # noqa: F401
from p3_veg_det.model import CenterNetDetector
from p3_veg_det.train import detection_panels, load_head
from p3_veg_det.train_eval import build_dataset, build_extractor, build_records, resolve
from p3_veg_det.viz import GT_COLOR, draw_boxes

VOLUME_ROOT = "/workspace/data/gwhd_2021"
#: What the training runs passed as data.root. It is in the compat hash, so load_head needs it
#: even though the images are read from the volume path here.
TRAINED_ROOT = "/root/data/gwhd_2021"
OUTPUTS = Path("/workspace/outputs/p3_veg_det")
CONFIGS = {  # arm label -> (config, output dir holding head_best.pt)
    "ConvNeXt-T s4": ("dev_convnext_s4", "convnext_s4_gauss_px"),
    "ConvNeXt-T s16": ("dev_convnext_s16", "convnext_s16_gauss_px"),
    "ViT-S+/16": ("dev_vit", "vit_gauss_px"),
}
assert list(CONFIGS) == list(ARMS), "arm order must match gwhd_figures.ARMS"

SPLIT_DOMAINS = {"train": "ETHZ_1", "val": "Usask_1", "test": "NAU_3"}
TEST_DOMAINS = ("UQ_11", "KSU_4", "NAU_3")


def typical(records: list, domain: str) -> int:
    """Index of the domain's image whose box count is closest to the domain median."""
    idx = [i for i, r in enumerate(records) if r.domain == domain]
    if not idx:
        raise SystemExit(f"no records for domain {domain!r}; have {sorted({r.domain for r in records})}")
    median = float(np.median([records[i].num_boxes for i in idx]))
    return min(idx, key=lambda i: (abs(records[i].num_boxes - median), records[i].image_name))


def split_examples(out: Path) -> None:
    config, _ = resolve("dev_convnext_s16", [f"data.root={VOLUME_ROOT}"])
    cells = []
    for split, domain in SPLIT_DOMAINS.items():
        records = build_records(config, split)
        record = records[typical(records, domain)]
        with Image.open(Path(VOLUME_ROOT) / "images" / record.image_name) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        drawn = Image.fromarray(draw_boxes(rgb, record.boxes, GT_COLOR, width=4))
        cells.append((f"{split} · {domain} · {record.num_boxes} heads", drawn))
    compose_grid([cells], out / "split_examples.jpg", tile=440)


def detections_test(out: Path) -> None:
    config, _ = resolve("dev_convnext_s16", [f"data.root={VOLUME_ROOT}"])
    records = build_records(config, "test")
    picks = [typical(records, d) for d in TEST_DOMAINS]
    print("test frames:", [(records[i].domain, records[i].image_name, records[i].num_boxes) for i in picks])

    by_arm = {}
    for label, (cfg, out_dir) in CONFIGS.items():
        trained, resolved = resolve(cfg, [f"data.root={TRAINED_ROOT}"])
        model = CenterNetDetector(
            build_extractor(trained, stub=False),
            kind=trained.model.head.kind,
            width=trained.model.head.width,
            amp_dtype=None,  # fp32 on CPU; see module docstring
            nms_kernel=trained.model.nms_kernel,
        )
        step = load_head(OUTPUTS / out_dir / "head_best.pt", model, resolved.compat_hash)
        dataset = build_dataset(config, records, model.stride, flip=False)
        panels = detection_panels(model, dataset, picks, "cpu", score_threshold=0.3)
        by_arm[label] = [(cap, Image.fromarray(rgb)) for rgb, cap in panels]
        print(f"{label}: step {step}", [cap for cap, _ in by_arm[label]])

    rows = [panel_row({label: panels[r] for label, panels in by_arm.items()})
            for r in range(len(picks))]
    compose_grid(rows, out / "detections_test.jpg")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    split_examples(out)
    detections_test(out)
    for path in sorted(out.glob("*.jpg")):
        print(f"wrote {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
