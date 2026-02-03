import random
import shutil
from pathlib import Path

# ====== YOUR SOURCE DATASET PATH ======
SOURCE_IMAGES_DIR = r"E:\FAISAL STUDIES\Vision Transformer Food Classification\images"

# ====== OUTPUT INSIDE YOUR PROJECT ======
OUTPUT_DIR = Path("data/food41_split")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    random.seed(SEED)

    src = Path(SOURCE_IMAGES_DIR)
    if not src.exists():
        print("ERROR: images folder not found:", src)
        return

    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    if not class_dirs:
        print("ERROR: images ke andar class folders nahi milay.")
        print("Expected: images\\apple_pie\\*.jpg (aapke screenshot jaisa)")
        return

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        images = [p for p in class_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not images:
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        def copy_to(split_name, split_imgs):
            out_class = OUTPUT_DIR / split_name / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                dst = out_class / img.name
                if dst.exists():
                    dst = out_class / f"{img.stem}_{random.randint(10000,99999)}{img.suffix}"
                shutil.copy2(img, dst)

        copy_to("train", train_imgs)
        copy_to("val", val_imgs)
        copy_to("test", test_imgs)

        print(f"{class_dir.name}: total={n}  train={len(train_imgs)}  val={len(val_imgs)}  test={len(test_imgs)}")

    print("\nDONE. Output folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
