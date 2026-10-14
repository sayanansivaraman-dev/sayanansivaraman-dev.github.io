---
title: Driving perception and VLM scene tagging on nuScenes (WIP)
subtitle: >
  Camera-only depth and bird's-eye-view perception on frozen DINOv3 features, plus a VLM for
  scene tagging.
description: >
  Can frozen vision foundation models carry camera-only driving perception, and can a VLM tag
  driving scenes well enough to slice the perception metrics?
meta:
  - label: Dataset
    value: nuScenes v1.0 — a subset of scenes with full sensor data, plus the map expansion.
  - label: VLM
    value: Qwen3.8-27B-FP8, served with vLLM
  - label: Backbones
    value: DINOv3 ViT-S+/16 and ConvNeXt-Tiny, both frozen, LVD-1689M pretraining. ResNet-50 baseline.
  - label: Stack
    value: PyTorch · vLLM · nuscenes-devkit · Weights & Biases · RunPod · Cloudflare R2
links:
  - label: DINOv3
    url: https://ai.meta.com/research/dinov3/
  - label: DINOv3 ViT-S+/16 on HuggingFace
    url: https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
  - label: DINOv3 ConvNeXt-Tiny on HuggingFace
    url: https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
  - label: Qwen3.8-27B-FP8 on HuggingFace
    url: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
  - label: nuScenes
    url: https://nuscenes.org/nuscenes
---

## Overview

**Personal Interests:** i) A refresher on perception for autonomous driving, with frozen DINOv3
features in place of trained backbones. ii) Hands-on with VLMs and their serving toolchain
(vLLM), a newer area for me. iii) Connecting the two: use the VLM's scene tags to slice the
perception metrics.

**Toolchain:** GitHub · HuggingFace (DINOv3, Qwen weights) · vLLM · nuscenes-devkit · Weights &
Biases · RunPod · Cloudflare R2 (dataset transfer)

**Dataset:** [nuScenes](https://nuscenes.org/nuscenes) v1.0 with the map expansion pack. Six
cameras, a 32-beam LiDAR, ego poses and calibration, across Boston and Singapore.

{% comment %}
[[Dataset numbers for the slice actually used: scenes, keyframes, sweeps, cities, train/val]]
{% endcomment %}

> **Licence.** nuScenes is CC BY-NC-SA 4.0: fine for a personal portfolio, and every published
> image on this page is credited to nuScenes.

{% comment %}
[[Hero figure: the six camera views of one sample, arranged around the ego vehicle]]
{% endcomment %}

---

## 1. Questions

How well do frozen DINOv3 features work for metric depth from
a single camera, six cameras fused into one bird's-eye view, and objects that matter at long range, as well as lanes, crosswalks, etc.


| # | question | arms | measured by |
|---|---|---|---|
| 1 | Do frozen DINOv3 features carry metric depth, and which backbone carries more? | ViT-S+/16 vs ConvNeXt-T s16; LiDAR vs map ground truth | depth error by distance band |
| 2 | Are frozen DINOv3 backbones viable for bird's-eye view? | ViT-S+/16 and ConvNeXt-T s16 (frozen) vs ResNet-50 (fine-tuned) | IoU per map layer and for vehicles, by distance band |
| 3 | How much can a 2d map supply as ground truth? | map depth, map layers, mapped traffic lights | agreement with LiDAR; lights visible by range |
| 4 | Can a general-purpose VLM tag driving scenes from a prompt alone? | Qwen3.8-27B-FP8, prompt only | precision / recall against nuScenes' scene descriptions |
| 5 | Do the VLM's tags explain where perception fails? | depth and BEV error, sliced by tag | error by tag vs by nuScenes' own descriptions |

Question 1 informs question 2: Lift-Splat-Shoot places image features in the bird's-eye view
using predicted depth, so poor frozen-feature depth would sink it. Published work is encouraging --
frozen DINOv2 in Lift-Splat-Shoot improved vehicle IoU on nuScenes by 7.4 points
([Hayes et al., 2025](https://arxiv.org/abs/2501.08118v1)) -- but took depth from a separate
model rather than from the DINO features.

We plan to get depth using DINOV3 features, adding a trainable head.

{% comment %}
[[What carries over from RELLIS-3D (stride, BEV fusion) and GWHD (domain shift, small objects)]]
{% endcomment %}

---

## 2. The dataset

Planned.

### A subset of scenes, with full data

{% comment %}
[[Why a subset: frozen features need fewer distinct scenes; full data per scene keeps dense,
  temporal and multi-sweep LiDAR training possible. Full data ships as 10 fixed parts.]]

[[Table: the scene mix of the parts downloaded -- city, night, rain, train/val -- checked against
  the metadata archive]]

[[Table: task | dense training? | how -- depth (every LiDAR sweep), map BEV (every frame),
  object BEV (keyframes only)]]

[[Note: dense frames are not independent samples; evaluation is per scene / on keyframes]]
{% endcomment %}

### Splits

{% comment %}
[[Official scene-level splits kept, so no frame leakage. The Boston → Singapore domain-shift
  option (left-hand traffic), analogous to GWHD's continent split]]
{% endcomment %}

### Transfer and storage

{% comment %}
[[Pipeline: v1.0-mini on the Mac → download on a CPU pod → one R2 object per part, unextracted →
  pull and extract onto each GPU pod's local disk. Sizes and monthly R2 cost.]]
{% endcomment %}

---

## 3. Monocular depth on frozen DINOv3

Planned.

{% comment %}
[[Setup: depth head on a frozen DINOv3 trunk, six cameras, SILog loss with λ = 0.85 so the head
  learns metric scale (Lift-Splat-Shoot needs it). Backbone run online, not cached -- table of
  per-sample feature sizes.]]
{% endcomment %}

### Ground truth from LiDAR

{% comment %}
[[Sparse 32-beam projection, loss only where a return exists, multi-sweep aggregation with
  ego-motion compensation, moving objects handled with the keyframe boxes]]
{% endcomment %}

### Ground truth from the map

{% comment %}
[[The map's layers are 2D (only traffic lights carry a height), so map depth = camera ray ∩ a
  ground surface fitted from LiDAR ground returns]]

[[Figure: map-depth error vs LiDAR, by distance -- and the range cap it justifies]]

[[Occlusion check: ray-cast against aggregated LiDAR plus keyframe boxes; what still slips
  through]]
{% endcomment %}

### Results

{% comment %}
[[Table: backbone | stride | GT source | AbsRel | RMSE | δ<1.25 -- by distance band]]

[[Figure: predicted depth vs projected LiDAR on a few frames]]

[[Optional: temporal self-supervision (Monodepth2-style, known ego pose), if run]]
{% endcomment %}

---

## 4. Traffic lights from the map

Planned.

{% comment %}
[[What nuScenes has: no traffic-light boxes, and no light state anywhere. The map expansion
  stores each light's 3D pose and bulb list -- table of lights per map, height quality by city]]

[[Figure: mapped lights projected into a camera frame, checking height and yaw conventions]]

[[Camera resolution vs range: front-camera focal length, pixels per housing at 50 / 100 / 200 m,
  and whether patch-level depth is viable at 200 m]]

[[Count of mapped lights in view by range, before and after the occlusion check]]

[[Result: far-range light detection with a per-light range output, if the counts justify it]]
{% endcomment %}

---

## 5. VLM scene tagging

Planned.

{% comment %}
[[Setup: Qwen3.8-27B-FP8 on vLLM -- GPU memory needed (weights ~28-30 GB at FP8), offline batch
  inference, JSON-schema structured outputs, prefix caching across the six cameras]]

[[The tag schema: night, rain, construction zone, intersection, occlusion, glare, ... and the
  prompt]]

[[Table: tag | precision | recall -- scored against nuScenes' human scene descriptions and object
  attributes]]

[[Throughput: samples / s, GPU, cost per 1k samples]]

[[Figure: example frames with their tags, including a few failures]]
{% endcomment %}

---

## 6. Slicing perception by scene tag

Planned.

{% comment %}
[[The result that connects both halves: depth error (and later BEV IoU) sliced by VLM tag --
  night vs day, rain vs dry. Compare against the same slicing using nuScenes' own descriptions.]]
{% endcomment %}

---

## 7. Bird's-eye view: Lift-Splat-Shoot from six cameras

Planned.

{% comment %}
[[Setup: lift each camera's features with the predicted depth, splat into a BEV grid, predict map
  layers -- drivable area, lane boundaries, crosswalks, stop lines -- rasterised from the map
  expansion around the ego pose]]

[[Backbone comparison, same Lift-Splat-Shoot head, data and schedule:
  DINOv3 ViT-S+/16 frozen | DINOv3 ConvNeXt-T s16 frozen (stages 2-3), both at stride 16 |
  ResNet-50 fine-tuned (comparability with the Lift-Splat-Shoot / BEVDet literature).
  Follow-up: ConvNeXt-T s8 (stages 1-3) if small / far objects lag.
  Table: backbone | IoU per map layer | vehicle IoU -- by distance band (0-20 / 20-35 / 35-50 m)]]

[[Prior work: frozen DINOv2 in Lift-Splat-Shoot (Hayes et al., 2025); DINOv2 + LoRA in SimpleBEV
  for robustness (Barın et al., https://arxiv.org/html/2409.10228v1). If frozen lags on
  detection: LoRA or unfreezing the last blocks]]

[[Table: layer | IoU -- camera-only vs a LiDAR control]]

[[Boston → Singapore: the domain-shift result]]

[[Temporal variant on the dense scene subset]]

[[Figure: BEV predictions vs map ground truth]]
{% endcomment %}

---

## 8. Further VLM experiments

Planned, ranked by how well each can be scored against nuScenes ground truth.

{% comment %}
[[Spatial reasoning against nuScenes boxes: counting and distance questions, accuracy by
  distance / size / occlusion (check NuScenes-QA and DriveLM first)]]

[[Rare-case mining: emergency vehicles, animals, construction -- retrieval precision / recall]]

[[Traffic-light state on projected map crops, scored against a few hundred hand labels]]

[[Distilling the VLM's tags into a small head on frozen DINOv3: agreement, latency, cost]]

[[Perception output as VLM context: do hallucinated objects drop?]]
{% endcomment %}

---

## 9. Learnings

In progress.

{% comment %}
[[One bullet per finding that changed the work, in the GWHD format: what looked fine, what the
  metric showed, what fixed it]]
{% endcomment %}
