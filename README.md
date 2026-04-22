# DepthAR

**Semantics-Aware Real-Time AR Occlusion via Neurosymbolic Depth Fusion from a Monocular Webcam**

DepthAR achieves physically correct AR occlusion from a single commodity webcam with no depth sensor. Virtual objects appear behind real-world foreground objects based on estimated scene depth — something standard webcam AR systems cannot do.

---

## Demo

[Insert demo video URL here]

---

## How It Works

DepthAR runs a four-stage pipeline on each webcam frame:

1. **Depth estimation** — Depth Anything V2 produces a per-pixel relative depth map from the RGB input
2. **Semantic segmentation** — SegFormer-B0 produces a binary person mask (stages 1 and 2 run in parallel)
3. **Symbolic correction** — rule-based boundary overrides fix depth errors at object edges using semantic priors
4. **Temporal stabilization** — an 8-frame EMA buffer eliminates frame-to-frame flickering

The symbolic correction layer is the core contribution: depth and semantics fail at different locations, and combining them via explicit rules produces cleaner occlusion masks than either model alone.

---

## Installation

```bash
git clone https://github.com/gargi-09/depthar
cd depthar
pip install -r requirements.txt
```

Python 3.9+ recommended. Runs fully on CPU — no GPU required.

---

## Usage

```bash
python tests/test_depth.py
```

**Controls:**

| Key | Action |
|---|---|
| `W / S` | Move object forward / back |
| `A / D` | Move object left / right |
| `I / K` | Move object up / down |
| `T` | Toggle temporal smoothing |
| `TAB` | Toggle debug view |
| `Q` | Quit |

---

## Project Structure

```
depthar/
├── core/
│   ├── depth_estimator.py       # Depth Anything V2 wrapper
│   ├── segmentor.py             # SegFormer-B0 wrapper
│   ├── occlusion_reasoner.py    # Symbolic correction layer
│   └── temporal_smoother.py    # EMA buffer
├── ar/
│   ├── renderer.py              # AR compositing and overlays
│   └── face_anchor.py           # Face-anchored reticle
├── tests/
│   └── test_depth.py            # Main entry point
└── requirements.txt
```

---

## Results

| Metric | Value |
|---|---|
| Temporal variance reduction | 5.1× (smoothing ON vs OFF) |
| Mode detection accuracy | 91% (200 labeled frames) |
| Runtime (CPU, no GPU) | 8–12 FPS |

---

## Future Work

- GPU deployment via Google Cloud Run for 30+ FPS
- Optical flow propagation for temporally consistent mask tracking
- Quantitative benchmarking against NYU Depth V2 and ScanNet

---

## References

- Yang et al., "Depth Anything V2," NeurIPS 2024
- Xie et al., "SegFormer," NeurIPS 2021
- Godard et al., "Monodepth2," ICCV 2019

---

## Acknowledgements

Built for CS 5330 Computer Vision at Northeastern University. Thanks to Prof. Bruce Maxwell and the course staff for guidance throughout the project.
