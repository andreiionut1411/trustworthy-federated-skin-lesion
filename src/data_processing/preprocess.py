import os
import cv2
import argparse
from tqdm import tqdm

def pad_resize(image, size=256):
    h, w, _ = image.shape
    scale = size / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create padded canvas
    canvas = cv2.copyMakeBorder(
        resized,
        top=(size - new_h) // 2,
        bottom=(size - new_h) - (size - new_h) // 2,
        left=(size - new_w) // 2,
        right=(size - new_w) - (size - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return canvas

def process_dataset(img_dir, size=256):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_name in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = pad_resize(img, size)

        # Overwrite original image
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Preprocessing complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    process_dataset(args.images, size=args.size)
