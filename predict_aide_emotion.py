import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def weighted_f1(y_true, y_pred, labels):
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    weighted = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        weighted += (support.get(label, 0) / total) * f1
    return weighted


class ClipImageAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, num_classes, num_prompts):
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.use_prompt_weight = True
        self.use_class_temperature = True
        self.use_class_bias = True

    def _adapt_image(self, image_x):
        base = self.input_proj(image_x)
        delta = self.net(base)
        fused = base + delta
        image = self.out_proj(fused)
        return image / image.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def logits(self, image_x, text_x):
        image = self._adapt_image(image_x)
        text = text_x / text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarity = torch.einsum("bd,cpd->bcp", image, text)
        if self.use_prompt_weight:
            prompt_weight = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_similarity = (similarity * prompt_weight).sum(dim=-1)
        else:
            class_similarity = similarity.mean(dim=-1)
        global_scale = self.logit_scale.exp().clamp(max=100.0)
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_similarity * class_scale + class_bias


def load_adapter_weights(adapter, state, device):
    adapter.input_proj.load_state_dict(state["input_proj"])
    adapter.net.load_state_dict(state["net"])
    adapter.out_proj.load_state_dict(state["out_proj"])
    adapter.logit_scale.data.copy_(state["logit_scale"].to(device))
    adapter.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(device))
    adapter.class_logit_scale.data.copy_(state["class_logit_scale"].to(device))
    adapter.class_bias.data.copy_(state["class_bias"].to(device))
    adapter.use_prompt_weight = state.get("use_prompt_weight", True)
    adapter.use_class_temperature = state.get("use_class_temperature", True)
    adapter.use_class_bias = state.get("use_class_bias", True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run minimal AIDE emotion inference")
    parser.add_argument("--dataset", default="demo_dataset/manifest.json")
    parser.add_argument("--checkpoint", default="assets/h2048_d02.ckpt.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="predictions.json")
    return parser.parse_args()


def load_clip(model_id, device):
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, local_files_only=True, use_safetensors=False)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


def build_text_features(class_labels, class_prompts, processor, model, device):
    all_features = []
    for label in class_labels:
        prompts = class_prompts[label]
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_features.append(text_features.float())
    return torch.stack(all_features, dim=0)


def build_image_feature(frame_paths, processor, model, device):
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    pooled = image_features.mean(dim=0, keepdim=True)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset_path = (root / args.dataset).resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    output_path = (root / args.output).resolve()

    with dataset_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_labels = manifest["class_labels"]
    class_prompts = manifest["class_prompts"]
    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"

    processor, clip_model = load_clip(config["model_id"], device)
    text_features = build_text_features(class_labels, class_prompts, processor, clip_model, device)

    adapter = ClipImageAdapter(
        dim=int(text_features.shape[-1]),
        hidden_dim=int(config["adapter_hidden_dim"]),
        dropout=float(config["adapter_dropout"]),
        num_classes=len(class_labels),
        num_prompts=int(text_features.shape[1]),
    ).to(device)
    load_adapter_weights(adapter, checkpoint["adapter_state_dict"], device)
    adapter.eval()

    predictions = []
    y_true = []
    y_pred = []
    for sample in manifest["samples"]:
        frame_paths = [str((dataset_path.parent / frame).resolve()) for frame in sample["frame_paths"]]
        image_feature = build_image_feature(frame_paths, processor, clip_model, device)
        with torch.no_grad():
            logits = adapter.logits(image_feature.float().to(device), text_features.to(device))
            pred_index = int(logits.argmax(dim=-1).item())
        predicted_label = class_labels[pred_index]
        predictions.append(
            {
                "sequence_id": sample["sequence_id"],
                "label": sample["label"],
                "prediction": predicted_label,
                "frame_paths": sample["frame_paths"],
                "label_prompts": sample["label_prompts"],
            }
        )
        y_true.append(sample["label"])
        y_pred.append(predicted_label)

    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / len(y_true) if y_true else 0.0
    result = {
        "dataset": manifest["name"],
        "num_samples": len(predictions),
        "accuracy": round(accuracy, 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, class_labels), 6),
        "predictions": predictions,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"accuracy": result["accuracy"], "weighted_f1": result["weighted_f1"], "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
