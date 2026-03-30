# AIDE+


This package contains a minimal inference-only version of the best AIDE emotion model.

## Contents

- `predict_aide_emotion.py`: offline inference script
- `assets/h2048_d02.ckpt.pt`: best frozen-CLIP adapter checkpoint
- `demo_dataset/manifest.json`: demo dataset manifest with labels and prompts
- `demo_dataset/clips/...`: sampled frames for each demo clip

## Environment

Install the pinned packages in `requirements.txt`.

## Run

```bash
python predict_aide_emotion.py --device cuda:0
```

If CUDA is not available:

```bash
python predict_aide_emotion.py --device cpu
```

The script writes predictions to `predictions.json`.
