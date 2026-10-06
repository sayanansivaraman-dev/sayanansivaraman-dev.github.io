---
title: Off-road vegetation perception on RELLIS-3D
subtitle: >
  Frozen DINOv3 features, lifted into a BEV LiDAR grid, then aggregated over time.
description: >
  Use pre-trained vision foundation models as a basis for perception in a new domain.
og_image: /assets/rellis3d/camera_coverage.png
meta:
  - label: Dataset
    value: RELLIS-3D — 13,556 synchronised camera/LiDAR frames, off-road.
  - label: Backbones
    value: DINOv3 ViT-S+/16 and ConvNeXt-Tiny, both frozen, LVD-1689M pretraining.
  - label: Stack
    value: PyTorch · ONNX · TensorRT · Weights & Biases · RunPod
  - label: Compute
    value: 1× RTX PRO 4500 Blackwell, ~$8 of GPU time
links:
  - label: DINOv3
    url: https://ai.meta.com/research/dinov3/
  - label: DINOv3 ViT-S+/16 on HuggingFace
    url: https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
  - label: DINOv3 ConvNeXt-Tiny on HuggingFace
    url: https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
  - label: RELLIS-3D
    url: https://github.com/unmannedlab/RELLIS-3D
---

## Overview

**Personal Interests:** i) Explore SOTA vision foundation models (DINOv3). ii) Applied to a domain new to me: offroad/agricultural robotics. iii) Fusion into 3D and a temporal model. iv) Claude-assisted rapid prototyping.

**Toolchain:** GitHub · HuggingFace (DINOv3 weights) · Weights & Biases · RunPod (1× RTX PRO 4500
Blackwell, ~$0.72/hr) · Cloudflare R2 (cache transfer) · Onnx-runtime · TensorRT

**Dataset:** [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) — off-road multimodal, 13,556
synchronised camera/LiDAR frames, 6,234 annotated images, per-point LiDAR annotations.

> **Dataset Limitations.** RELLIS-3D is a small academic dataset. Perfect for a side-project like this, but not necessarily the scale needed to ship a real product.

>RELLIS-3D's official splits are by
> **frame, not sequence**, so there is a sequence temporal overlap, otherwise known as dataset leakage. This is a common limitation for small, academic datasets. Roughly 47% of val >and 69% of test frames come from dataset collections
> that also appear in train. 

> **Sequence 00001 is the only train-disjoint
> traversal** — 1,047 test frames and 1,272 val frames, one contiguous cut at frame 1047. We 
> report metrics on this sequence, disjoint from the training set.

---

## 1. Image segmentation: a linear probe on frozen DINOv3

**Question.** Can we use a small 1x1-conv (27k-parameter) linear probe on top frozen
[DINOv3](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) features to get solid image-segmentation accuracy on an agricultural off-road dataset?

Take the smallest ConvNeXt model (ConvNeXt-Tiny) and similarly sized ViT-S+/16 from the DINOv3 HuggingFace model repo, keep the backbone parameters frozen, and train a 1x1-conv layer on top, for image-segmentation. We conducted a few experiments, and the best-performing model came within striking distance of the top published results on the RELLIS-3D dataset.


![frozen DINOv3 probe predictions]({{ '/assets/rellis3d/segmentation_panel.png' | relative_url }})

*The most accurate image-segmentation model (ConvNeXt-s4, CE+Dice, 0.53760 val mIoU) on two
held-out validation frames from sequence 00001.*

*Magenta marks every pixel where the prediction disagrees with a non-void label. The errors are
generally on the boundary: grass/bush, bush/tree canopy, the rubble piles.*

Both backbones are the LVD-1689M pretrained releases, used entirely frozen:
[ViT-S+/16](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) and
[ConvNeXt-Tiny](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m). Only the
1×1 probe trains.

| backbone | stride | channels | probe params | best @ | best val mIoU | final | gap |
|---|---|---|---|---|---|---|---|
| ViT-S+/16 | 16 | 1536 | 29,203 | 39% | 0.50736 | 0.48324 | +0.0241 |
| ConvNeXt-T, layers [2,3] | 16 | 1152 | 21,907 | 91% | 0.48198 | 0.45708 | +0.0249 |
| **ConvNeXt-T, layers [0,1,2,3]** | **4** | 1440 | **27,379** | 75% | **0.51982** | 0.48827 | +0.0316 |

Pre-trained models tend to overfit rapidly, and we are only training the 1x1 conv head. We select the best checkpoint based on its accuracy on the validation set. "Train longer" only applies to training from scratch.

The ViT's best checkpoint lands fairly early, 39% into its run, ConvNeXT s4's at 75%, s16's at 91% 

We find that a finer stride achieves better accuracy. The multi-scale stride-4 features beat stride-16 by **+0.038 mIoU**. The finer stride led to the model we used going forward.

We run the best configuration of the held-out test-set. It achieves **0.46577 test mIoU** over 18 classes — within 2–4 points of fully-trained segmentation networks
published on this dataset (HRNet+OCR 48.83, GSCNN 50.13).

**Loss-function sweep**, ConvNeXt-s4:

| objective | best val mIoU | final | gap |
|---|---|---|---|
| **CrossEntropy + Dice** | **0.53760** | 0.52499 | **+0.0126** |
| Focal + Dice | 0.53593 | 0.52120 | +0.0147 |
| CrossEntropy | 0.51982 | 0.48827 | +0.0316 |
| Focal | 0.51180 | 0.47211 | +0.0397 |
| BinaryCrossEntropy + Dice | 0.45532 | 0.44641 | +0.0089 |

The BinaryCrossEntropy loss performed the worst. Focal and "vanilla" CrossEntropy loss performed similarly.

Adding a Dice loss term reduces overfitting and improved overall accuracy. This is the loss configuration retained for the subsequent steps.

---

## 2. Lifting to 3D: BEV fusion

**How much does adding camera data help 3d segmentation accuracy?** RELLIS-3D has only a
forward-facing camera, so the honest version of the question is narrower: how much does it help
*where it actually sees*, and what does carrying it cost everywhere else?

Grid: 256×256 cells at 20 cm (±25.6 m), 12 vertical levels at 50 cm. Each vertical level has mean-intensity of the lidar points, and the raw count of points. We also include an x-plane, and y-plane, so the network knows the (x, y) of the cell centers.

| arm | channels | composition |
|---|---|---|
| C1 | 26 | 24 geometry + 2 coordinate planes |
| C2 | 64 | C1 + 38 camera (19 classes × 2 vertical bands) |

The BEV-Tower is a 1.92 M-parameter U-Net. The camera variant (C2) has 38 additional input channels, for the camera inputs.

We project the lidar pointcloud into the camera view; the corresponding BEV cells get that camera-pixel's segmentation probabilities across the 19 categories. Multiple pixels projecting into the same BEV Gridcell are simply averaged. The input to the BEV U-Net has a BatchNorm2D.

Only 12.2% of scored cells carry a LiDAR return that also lands in the camera image (val 12.2%,
test 12.3%). 

To measure the camera's contribution we score C1 and C2 on the **same cells**, split by whether
the camera reached them. The coverage mask comes from camera geometry, not from either model, so
the two arms are scored on identical populations — and the uncovered slice becomes a control: if
C2 moves cells the camera never saw, the gain is not the camera's.

| block | slice | C1 fwIoU | C2 fwIoU | Δ |
|---|---|---|---|---|
| val | covered | 0.5589 | 0.6184 | **+0.0594** |
| val | uncovered | 0.4239 | 0.4113 | −0.0126 |
| val | whole grid | 0.4264 | 0.4205 | −0.0059 |
| test | covered | 0.7027 | 0.7334 | **+0.0307** |
| test | uncovered | 0.5029 | 0.4902 | −0.0127 |
| test | whole grid | 0.5170 | 0.5082 | −0.0088 |

**The camera helps where it sees, and costs a little where it doesn't.** The covered gain
replicates across both the disjoint validation and test sets, and so does the uncovered cost: −0.0126 and −0.0127, agreeing to four decimals. That cost is not noise: the 38 camera channels are zero on 88% of the grid, and the tower spends capacity on them anyway.

Slicing by covered vs. uncoveraged allows us to measure where the camera improves accuracy. Reporting just the `whole grid` without slicing would appear to show almost no benefit to using the camera.

![camera coverage]({{ '/assets/rellis3d/camera_coverage.png' | relative_url }})

*Where the two sensors overlap, over 400 val frames. "Covered" means a cell got a LiDAR return that
projects into the image.*

| band | scored cells/frame | covered/frame | covered ÷ scored |
|---|---|---|---|
| 0–2 m | 0.0 | 0.0 | — |
| 2–4 m | 464 | 4 | 0.9% |
| 4–6 m | 1,506 | 133 | 8.9% |
| 6–18 m | 2,208 → 1,556 | 243 → 190 | 11.0–12.2% |
| 18–26 m | 1,256 → 488 | 167 → 96 | 13.3–19.7% |

The camera's appearance cues help where geometry is ambiguous. Per-level occupancy for the two largest classes, split by vertical (Z-coordinate):

| class | −1.5 m | −1.0 m | −0.5 m | 0.0 m |
|---|---|---|---|---|
| grass | **62.4%** | 25.2% | 4.2% | 1.0% |
| bush | 36.2% | **51.1%** | 32.8% | 11.6% |

`grass` peaks on the ground; `bush` peaks one 50 cm band higher, but 36.2% of bush cells still
carry a ground return. This is a source of easy confusion, as the height is the strongest feature to distinguish these two categories.

![grass vs bush occupancy]({{ '/assets/rellis3d/occupancy_profile.png' | relative_url }})

---

## 3. Temporal aggregation

We created a temporal model by pose-aligning the 4 most-recent BEV+camera (single-frame) predictions with the current prediction. The top-line metric, mIoU barely moves. We added a temporal stability metric to surface benefits of temporal aggregation.

### A flicker metric

Warp frame *k−1*'s prediction into frame *k* using the relative pose, take the argmax of both, and
count the cells whose class changed. Scored on cells that are labelled in *k* and had warp support —
the same population mIoU scores, so the two numbers refer to the same cells.

We tested 4 different fusion heads, with increasing non-linearity, spatial, and temporal context.

### Fusion Heads for Temporal Aggregation

| head | params | linear? | per-cell? | sees history? |
|---|---|---|---|---|
| `conv1x1` | 1,140 | yes | yes | yes |
| `mlp` | 5,839 | no | yes | yes |
| `unet` | 36,655 | no | **no** | yes |
| `unet` control | 36,655 | no | no | **no** |

The per-cell heads are tiny and feed-forward, without additional spatial context. The U-Net is the first head with a
spatial receptive field, which is the one thing no per-cell head can have at any width.

As a control, we also ran a U-Net with 5 identical inputs — `ablate_history` fills all four history slots with a copy of
the *current* frame, so the head sees the same tensor shape and the same parameter count with
none of the information.

We use the following accuracy metrics.

- **mIoU** — the unweighted mean of per-class IoU. Every class counts equally, so `rubble` (26
  cells) carries the same weight as `bush` (6.29 M).
- **fwIoU** — the same per-class IoUs, each weighted by that class's share of labelled cells, so
  the score follows the bulk of the scene instead of the tail.

Temporal aggregation moves mIoU by +0.0002 — nothing — and fwIoU by **+0.0173**.

| arm | mIoU | fwIoU | flicker |
|---|---|---|---|
| single-frame C2 | 0.41413 | 0.4205 | 0.2001 |
| `conv1x1` temporal | 0.39689 | — | — |
| `mlp` temporal | 0.41148 | 0.4313 | 0.2227 |
| `unet` control | 0.41538 | 0.4095 | 0.2036 |
| **`unet` temporal** | 0.41437 | **0.4378** | **0.1696** |


### Two Highlighted Observations

Temporal aggregation cuts flicker ~16%. Against the matched control: 0.0340 on val (16.7%),
0.0314 on test (15.7%).

The gain grows with distance from sensor.

![flicker by range]({{ '/assets/rellis3d/flicker_by_range.png' | relative_url }})

### Loss function: an accuracy/stability tradeoff

Adding Dice loss helps accuracy in the temporal BEV U-Net, just as it did for the image-only model. Interestingly, adding Dice loss increases flicker.  Dice commits harder on sparse evidence: better per frame, more volatile between frames.

| | CE | CE+Dice | Δ |
|---|---|---|---|
| val fwIoU | 0.4378 | 0.4526 | **+0.0148** |
| test fwIoU | 0.5228 | 0.5342 | **+0.0114** |
| val flicker | 0.1696 | 0.1894 | +0.0199 |
| test flicker | 0.1687 | 0.1875 | +0.0187 |

### Category Confusion

We break this down further by looking at the confusion matrix.

![confusion matrix]({{ '/assets/rellis3d/confusion.png' | relative_url }})

*The temporal U-Net on **CE+Dice** (step 500, test block) — the best arm, and the fourth bar on
the right. Row-normalised, so the diagonal is recall. Support is in each row label. `water` (101
cells) and `rubble` (26) are greyed: at ~1e-6 of the block their rates are sampling noise, not
measurement. Columns are the labelled classes only — the model also emits `barrier`, `fence` and
`log`, which never appear in this block's labels, but that is 0.075% of predictions.*

`grass` is a sink. Nearly every minority class bleeds into it: mud 77%, water 57%, bush 49%,
puddle 47%, concrete 44%, person 39% — while `grass` itself holds 0.909 recall. Geometrically, any other category on the ground is likely to confuse with grass. Further, grass tends to visually and physically overlap with other categories. This is apparent in the image view.

`tree` holds 0.921 recall and sends just 1.4% to grass — alone among the non-grass classes. It is vertically separated from grass, so geometry helps a lot.

`bush` the single largest class at 6.29 M cells, with a recall of 0.483. Confusion bush -> grass is the largest error term.

The right panel shows how the bush->grass confusion improves with various model configurations.
The ablated control is **worst** (0.551), single-frame BEV+camera sits between
(0.532), temporal aggregation (CrossEntropy loss) improves it (0.510), and CE+Dice improves it again (0.486) — a 12%
relative reduction from control to best. History is doing real work on accuracy, not just stability.

> The `bev_cam` arm is scored on 1,047 frames against the temporal arms' 1,043: the temporal head
> needs four predecessors, so it drops the block's first four frames. A 0.4% difference in
> population, noted rather than hidden.

### Camera coverage and stability

Flicker sliced by camera coverage — covered in both scans, or neither. Unet CE, val block:

| slice | flicker | cells |
|---|---|---|
| covered in both | **0.0627** | 1,087,317 |
| covered in neither | 0.1758 | 12,672,966 |

**2.80× steadier where the camera sees the cell in both frames.** Running the identical metric on the ground-truth labels gives the same split, 2.76×.
Covered cells are nearer, denser and more often revisited, which makes them easier to annotate and predict consistently.

![the temporal arm]({{ '/assets/rellis3d/temporal_cam_unet.gif' | relative_url }})

*Row 1: camera and raw LiDAR height. Row 2: image segmentation and fused BEV prediction. Row 3:
ground truth. Sequence 00001, consecutive scans.*

---

## 4. Deployment: Torch → ONNX → TensorRT

We export the torch model using onnx-runtime, then benchmark the model using the tensor-rt inference engine. Specs: RTX PRO 4500 Blackwell / CUDA 13.0 / TRT 11.3. In constructing the deploy-graph, we lean on the fact that the 4 single-frame results from the previous timesteps need not be re-computed. 

| runtime | precision | median | vs torch | max \|Δ\| vs CPU fp32 |
|---|---|---|---|---|
| torch | TF32 conv (default) | 10.51 ms | — | — |
| onnxruntime | CPU fp32 | — | — | 5e-05 |
| TensorRT | fp32 strict | 7.24 ms | 1.45× | 6.6e-05 |
| TensorRT | tf32 | **4.77 ms** | **2.20×** | 2.0e-02 |

Notes: We need ONNX opset version >=17, as this supports LayerNorm, which is required by the ConvNeXT models.

---

## 5. Learnings

 - This was a pretty fun and cheap little project to spin up.
 - The initial goal was to explore the use of foundation models for new vision/perception tasks.
 - The project tied together toolchains including W&B, Github, HuggingFace, RunPod, ONNX, and TensorRT.
 - Using Claude, I was able to generate most of the boilerplate within a week or two.
 - Training runs typically took 1hr or less.
