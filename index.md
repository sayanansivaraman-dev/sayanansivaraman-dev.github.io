---
title: Sayanan Sivaraman
layout: default
og_image: /assets/portrait.jpg
# A 600x991 portrait, so `summary` (small square-ish thumbnail) rather than the default
# `summary_large_image`, which crops to ~1.91:1 and would slice a horizontal band out of the
# middle of the photo. A large card also wants >=1200x630; this image is well under that.
og_card: summary
---

<div class="hero">
<img class="portrait" src="{{ '/assets/portrait.jpg' | relative_url }}"
     alt="Sayanan Sivaraman" />
<p class="intro">
I'm Sayanan. I've worked in Machine Learning, Software Engineering, and Perception for over a decade, most recently at Apple (2015-2026), and at Volkswagen before that (2013-2015). This page documents some of my personal/side projects.
</p>
</div>

## Recent Projects
{: #projects}

<div class="cards">

<a class="card" href="{{ '/projects/rellis3d/' | relative_url }}">
  <h3>Off-road vegetation perception on RELLIS-3D</h3>
  <p>Agricultural, off-road semantic segmentation in images and lidar, using the RELLIS-3D
    dataset.</p>
  <p>Frozen DINOv3 image models adapted for perspective segmentation, lifted into a BEV LiDAR grid
    with U-Net BEV backbone. Temporal aggregation using pose estimates and another, smaller BEV
    U-Net.</p>
  <p>Export model using ONNX and TensorRT.</p>
  <span class="tags">DINOv3 · BEV fusion · temporal aggregation · ONNX / TensorRT</span>
</a>

{% comment %}
  Planned write-ups, hidden until each has something to show. A LIQUID comment, not an HTML one:
  Liquid runs before kramdown, so this is stripped at build time and never reaches the browser.
  `<!-- -->` would still ship "Planned." in the page source, which announces an empty shelf.

  Uncomment one at a time as it lands, and keep it INSIDE `<div class="cards">` — kramdown parses
  raw HTML verbatim only when the outermost tag is block-level, so a card moved outside the
  wrapper gets its closing tag escaped into visible text.

  The `.pending` style (dashed border, dimmed, pointer-events: none) is still in style.css for
  when a card should be visible but not yet clickable.
{% endcomment %}
{% comment %}
<div class="card pending">
  <h3>GHWD — Global Wheat Head Detection</h3>
  <p>Planned.</p>
  <span class="tags">detection · agricultural imagery</span>
</div>

<div class="card pending">
  <h3>NuScenes</h3>
  <p>Planned.</p>
  <span class="tags">autonomous driving · multi-sensor</span>
</div>

<div class="card pending">
  <h3>Multimodal-Mind2Web</h3>
  <p>Planned.</p>
  <span class="tags">vision-language · agents</span>
</div>
{% endcomment %}

</div>

## About me
{: #about}

 - 11 years at Apple: 
    - 2 years in Siri (agentic harness, tool-calling, evaluation, simulation, on-device model, fine-tuning and distillation)
    - 9 years on Special Projects Group (perception, machine learning, computer vision).
  - 2 years at Volkswagen Group of America.
 - Looking to find my next role in robotics, machine learning, or modern agentic systems.
 

## Elsewhere

- [GitHub](https://github.com/sayanansivaraman-dev)
- [LinkedIn](https://www.linkedin.com/in/sayanan-sivaraman)
