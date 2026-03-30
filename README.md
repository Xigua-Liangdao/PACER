# PACER

## Appendix-Style Artifact Description

PACER is a compact, inference-only artifact for prompt-based driver emotion recognition from in-cabin driving clips. This repository is prepared as a lightweight release for paper appendix material, supplementary review, and artifact evaluation.

The package preserves the full prompt-conditioned inference path while removing training infrastructure, ablation scripts, and large-scale preprocessing components. The result is a minimal repository that can be inspected quickly and executed directly once the CLIP backbone has been cached locally.

---

## A. Artifact Scope

This repository contains only the components required to reproduce PACER inference:

- `predict_aide_emotion.py`
  - Main inference entrypoint.
  - Loads the CLIP backbone in offline mode.
  - Loads the released PACER adapter checkpoint.
  - Reads labels and prompt groups from a manifest file.
- `assets/h2048_d02.ckpt.pt`
  - Released checkpoint used by this artifact.
- `demo_dataset/manifest.json`
  - Self-contained manifest describing labels, prompts, and sample frame paths.
- `demo_dataset/clips/...`
  - Runnable demo subset with sampled in-cabin frames.
- `requirements.txt`
  - Minimal pinned dependencies.

This artifact does not include the full research codebase or the full AIDE dataset.

---

## B. Method Summary

PACER uses a frozen CLIP image-text backbone together with a lightweight adapter head.

At inference time, the procedure is:

1. Read class labels and prompt groups from the dataset manifest.
2. Encode prompt groups into normalized text prototypes.
3. Load the sampled in-cabin frames for each clip.
4. Encode each frame with CLIP and pool the frame features into a clip-level representation.
5. Refine the pooled visual embedding with the adapter head.
6. Compute similarity between the adapted image feature and the prompt-conditioned text features.
7. Output the highest-scoring emotion label.

The repository is intentionally structured so that prompt definitions remain in the manifest rather than in Python constants.

---

## C. Demo Subset Summary

The bundled demo package contains:

- 5 emotion classes
- 25 clips in total
- 5 clips per class
- 5 sampled frames per clip

Emotion classes:

- Anxiety
- Peace
- Weariness
- Happiness
- Anger

The included demo subset is intended for runnable inspection only. Full-scale evaluation should be performed with a full manifest following the same schema.

---

## D. Qualitative Examples

### D.1 Representative Frames by Class

| Anxiety | Peace | Weariness | Happiness | Anger |
|---|---|---|---|---|
| ![Anxiety example](demo_dataset/clips/2563/frames/0.jpg) | ![Peace example](demo_dataset/clips/0069/frames/0.jpg) | ![Weariness example](demo_dataset/clips/0823/frames/0.jpg) | ![Happiness example](demo_dataset/clips/1960/frames/0.jpg) | ![Anger example](demo_dataset/clips/2184/frames/0.jpg) |

### D.2 Additional Diversity Examples

| Anxiety | Peace | Weariness | Happiness | Anger |
|---|---|---|---|---|
| ![Anxiety sample 2](demo_dataset/clips/0195/frames/22.jpg) | ![Peace sample 2](demo_dataset/clips/1009/frames/22.jpg) | ![Weariness sample 2](demo_dataset/clips/1623/frames/22.jpg) | ![Happiness sample 2](demo_dataset/clips/2488/frames/22.jpg) | ![Anger sample 2](demo_dataset/clips/1500/frames/22.jpg) |

### D.3 Temporal Sampling Illustration

PACER performs clip-level inference rather than isolated single-frame classification. The following 5-frame example illustrates the temporal sampling pattern used for one demo clip.

| t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|
| ![Temporal frame 1](demo_dataset/clips/2563/frames/0.jpg) | ![Temporal frame 2](demo_dataset/clips/2563/frames/11.jpg) | ![Temporal frame 3](demo_dataset/clips/2563/frames/22.jpg) | ![Temporal frame 4](demo_dataset/clips/2563/frames/33.jpg) | ![Temporal frame 5](demo_dataset/clips/2563/frames/44.jpg) |

---

## E. Repository Layout

```text
PACER/
├── predict_aide_emotion.py
├── README.md
├── requirements.txt
├── assets/
│   └── h2048_d02.ckpt.pt
└── demo_dataset/
    ├── manifest.json
    └── clips/
        └── <sequence_id>/frames/*.jpg
```

---

## F. Environment And Dependencies

Recommended Python version:

- Python 3.8+

Pinned dependencies:

```text
torch==2.0.1
transformers==4.37.2
Pillow==8.4.0
```

Installation:

```bash
pip install -r requirements.txt
```

---

## G. Offline Model Requirement

PACER is configured to run with offline CLIP loading. Internally, the script uses local-only loading for both the processor and the model.

The required pretrained backbone is:

- `openai/clip-vit-base-patch32`

This model must already exist in the local Hugging Face cache before execution. If it is not yet cached, download it once in an online environment and reuse the local cache afterward.

---

## H. Running The Artifact

### H.1 Demo Inference on GPU

```bash
python predict_aide_emotion.py \
  --dataset demo_dataset/manifest.json \
  --checkpoint assets/h2048_d02.ckpt.pt \
  --device cuda:0
```

### H.2 Demo Inference on CPU

```bash
python predict_aide_emotion.py \
  --dataset demo_dataset/manifest.json \
  --checkpoint assets/h2048_d02.ckpt.pt \
  --device cpu
```

### H.3 Full-Dataset Inference

If a full manifest is available and follows the same schema, PACER can be applied directly without code changes.

```bash
python predict_aide_emotion.py \
  --dataset /path/to/full_dataset_manifest.json \
  --checkpoint assets/h2048_d02.ckpt.pt \
  --device cuda:0
```

### H.4 Custom Output Path

```bash
python predict_aide_emotion.py \
  --dataset /path/to/full_dataset_manifest.json \
  --checkpoint assets/h2048_d02.ckpt.pt \
  --device cuda:0 \
  --output /path/to/output_predictions.json
```

---

## I. Command-Line Arguments

- `--dataset`
  - Path to the manifest file.
  - Default: `demo_dataset/manifest.json`
- `--checkpoint`
  - Path to the PACER checkpoint.
  - Default: `assets/h2048_d02.ckpt.pt`
- `--device`
  - Device string such as `cuda:0` or `cpu`.
  - Default: `cuda:0`
  - If CUDA is requested but unavailable, the script falls back to CPU.
- `--output`
  - Output path for saved predictions.
  - Default: `predictions.json`

---

## J. Expected Manifest Schema

PACER expects the manifest to provide:

- `name`
- `class_labels`
- `class_prompts`
- `samples`
  - `sequence_id`
  - `label`
  - `label_prompts`
  - `frame_paths`
  - `source_emotion_label`

This design keeps prompts bundled with the dataset description, making the artifact easier to inspect and easier to transfer to full-scale evaluation data.

---

## K. Notes For Reviewers

- This repository is intentionally minimal and appendix-oriented.
- The checkpoint is fixed and stored locally in the repository.
- The demo subset is bundled only for lightweight verification.
- Full evaluation should use a complete manifest in the same schema.
- All naming in this artifact is aligned with PACER.

---

## L. Reproduction Checklist

Before execution, verify the following:

1. Python dependencies are installed.
2. `openai/clip-vit-base-patch32` is cached locally.
3. Manifest paths are valid on the local machine.
4. The checkpoint path points to a compatible PACER checkpoint.

Once these conditions hold, PACER can be executed directly without network access.

---

## M. Limitations

- This repository is not the full AIDE dataset release.
- This repository is not the full PACER research codebase.
- The artifact is intended for inference verification, not end-to-end training.
- Offline execution still requires a previously cached CLIP backbone.

---

## N. Citation

If this artifact is used in supplementary review or reproduction notes, please cite the associated PACER paper or project release.
