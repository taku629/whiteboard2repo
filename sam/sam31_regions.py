"""SAM 3.1 grounding pass for Whiteboard2Repo.

Runs Meta SAM 3.1 (promptable concept segmentation) on whiteboard photos and
writes `<name>.sam.json` (regions in 0-1000 coords) + `<name>.sam.png` overlay.
The web app feeds these regions to Muse Spark Open as grounding hints.

Runs on any CUDA GPU (we used a free Kaggle T4); CPU works with >=8 GB RAM.
    python sam/sam31_regions.py samples/board_shop.jpg samples/board_rooms.jpg
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

PROMPTS = {
    "rectangle box": "#1f5fd6",
    "cylinder": "#d97706",
    "cloud shape": "#7c3aed",
    "arrow": "#8a5cf6",
    "handwritten text": "#d23b3b",
}
MODEL_ID = "unsloth/sam3.1"  # SAM 3.1 detector in transformers Sam3Model format


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def main(paths: list[str]) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(MODEL_ID, dtype=dtype).to(dev).eval()
    proc = Sam3Processor.from_pretrained(MODEL_ID)
    for path in paths:
        if path.startswith("http"):
            local = Path(path.split("/")[-1])
            urllib.request.urlretrieve(path, local)
            path = str(local)
        img = Image.open(path).convert("RGB")
        W, H = img.size
        regions = []
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        for prompt, color in PROMPTS.items():
            inputs = proc(images=img, text=prompt, return_tensors="pt").to(dev)
            inputs = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            res = proc.post_process_instance_segmentation(
                out, threshold=0.35, mask_threshold=0.5,
                target_sizes=[[H, W]])[0]
            for box, score, mask in zip(res["boxes"].tolist(), res["scores"].tolist(), res["masks"]):
                b = [round(box[0] / W * 1000), round(box[1] / H * 1000), round(box[2] / W * 1000), round(box[3] / H * 1000)]
                if (b[2] - b[0]) * (b[3] - b[1]) > 500_000:  # whole-board false positives
                    continue
                if any(r["prompt"] == prompt and iou(r["bbox"], b) > 0.6 for r in regions):
                    continue
                regions.append({"prompt": prompt, "bbox": b, "score": round(score, 3)})
                m = Image.fromarray((mask.cpu().numpy() > 0).astype("uint8") * 90)
                rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                draw.bitmap((0, 0), m, fill=rgb + (110,))
                draw.rectangle(box, outline=color, width=4)
        stem = Path(path).stem
        Path(f"{stem}.sam.json").write_text(json.dumps({"model": "SAM 3.1", "image_size": [W, H],
                                                         "regions": regions}, indent=1))
        overlay.save(f"{stem}.sam.png")
        print(stem, len(regions), "regions")


if __name__ == "__main__":
    main(sys.argv[1:])
