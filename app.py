import json
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_DIR = Path("models/vit_food41")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

id2label = json.loads((MODEL_DIR / "id2label.json").read_text(encoding="utf-8"))

@torch.no_grad()
def predict(img: Image.Image):
    if img is None:
        return {}
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    topk = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:5]
    return {id2label[str(i)]: float(p) for i, p in topk}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction"),
    title="Faisal Sajjad — Vision Transformer Food Classifier",
    description="Fine-tuned ViT (vit-base-patch16-224). Dataset: your current 101 classes."
)

if __name__ == "__main__":
    demo.launch()
