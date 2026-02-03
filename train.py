# CHECKPOINT-ENABLED TRAIN.PY (Food-101/your current 101 classes)
import os
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

DATA_DIR = Path("data/food41_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

MODEL_NAME = "google/vit-base-patch16-224"
OUT_DIR = Path("models/vit_food41")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "checkpoint.pt"

EPOCHS = 3
BATCH_SIZE = 16
LR = 3e-5
NUM_WORKERS = 0
SEED = 42
USE_FP16 = True

torch.manual_seed(SEED)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return train_tfms, eval_tfms

@dataclass
class HFImageBatch:
    pixel_values: torch.Tensor
    labels: torch.Tensor

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    pixel_values = torch.stack(images, dim=0)
    return HFImageBatch(pixel_values=pixel_values, labels=labels)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for b in loader:
        pv = b.pixel_values.to(device, non_blocking=True)
        y  = b.labels.to(device, non_blocking=True)
        logits = model(pixel_values=pv).logits
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

def save_checkpoint(model, optimizer, scaler, epoch, best_val):
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scaler, device):
    if not CHECKPOINT_PATH.exists():
        return 1, 0.0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val", 0.0))
    print(f"[RESUME] checkpoint loaded -> next epoch {start_epoch} | best_val {best_val:.4f}")
    return start_epoch, best_val

def main():
    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_tfms, eval_tfms = build_transforms()
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds  = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    class_names = train_ds.classes
    num_labels = len(class_names)
    print("Classes:", num_labels)

    label2id = {n:i for i,n in enumerate(class_names)}
    id2label = {i:n for i,n in enumerate(class_names)}
    (OUT_DIR/"label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    (OUT_DIR/"id2label.json").write_text(json.dumps(id2label, indent=2), encoding="utf-8")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_FP16 and device.type == "cuda"))

    start_epoch, best_val = load_checkpoint(model, optimizer, scaler, device)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for b in pbar:
            pv = b.pixel_values.to(device, non_blocking=True)
            y  = b.labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_FP16 and device.type == "cuda")):
                out = model(pixel_values=pv, labels=y)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Val Acc: {val_acc:.4f}")

        save_checkpoint(model, optimizer, scaler, epoch, best_val)

        if val_acc > best_val:
            best_val = val_acc
            print("Saving best model...")
            model.save_pretrained(OUT_DIR)
            processor.save_pretrained(OUT_DIR)

    print("Testing best model...")
    best_model = ViTForImageClassification.from_pretrained(OUT_DIR).to(device)
    test_acc = evaluate(best_model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")
    print("Done. Model dir:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
