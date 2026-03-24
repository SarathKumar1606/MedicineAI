import easyocr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np

# -------------------------------
# Load EasyOCR (Detection)
# -------------------------------
reader = easyocr.Reader(['en'], gpu=False)  # set True if GPU

# -------------------------------
# Load TrOCR (Recognition)
# -------------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# -------------------------------
# Preprocessing (for handwriting)
# -------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=20)

    # Slight blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return img, gray


# -------------------------------
# Crop image using bbox
# -------------------------------
def crop_from_bbox(image, bbox):
    pts = np.array(bbox).astype(int)
    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])
    x_max = np.max(pts[:, 0])
    y_max = np.max(pts[:, 1])

    return image[y_min:y_max, x_min:x_max]


# -------------------------------
# TrOCR on cropped region
# -------------------------------
def trocr_on_crop(crop_img):
    if crop_img is None or crop_img.size == 0:
        return ""

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


# -------------------------------
# Hybrid OCR
# -------------------------------
def hybrid_ocr(image_path):
    original_img, processed = preprocess_image(image_path)

    # EasyOCR detection
    results = reader.readtext(
        processed,
        detail=1,
        low_text=0.2,
        text_threshold=0.3
    )

    # Sort top-to-bottom
    results = sorted(results, key=lambda x: x[0][0][1])

    final_texts = []

    for (bbox, text, prob) in results:
        crop = crop_from_bbox(original_img, bbox)

        # Decide when to use TrOCR
        if prob < 0.5 or len(text) < 3:
            trocr_text = trocr_on_crop(crop)

            if trocr_text.strip():
                final_texts.append(trocr_text)
            else:
                final_texts.append(text)
        else:
            final_texts.append(text)

    return "\n".join(final_texts)





"""
import argparse
import os
from src.ocr import hybrid_ocr
from src.matcher import match_medicines


def run_pipeline(image_path, output_dir="outputs/predictions"):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📄 Processing: {image_path}")

    # ---- Step 1: OCR ----
    print("\n🔍 Running OCR...")
    extracted_text = hybrid_ocr(image_path)

    print("\n----- OCR TEXT -----\n")
    print(extracted_text)

    # ---- Step 2: Medicine Matching ----
    print("\n💊 Matching medicines...")
    medicines = match_medicines(extracted_text)

    print("\n----- MATCHED MEDICINES -----\n")
    for med in medicines:
        print(f"- {med}")

    # ---- Step 3: Save Output ----
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, filename + ".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("OCR TEXT:\n")
        f.write(extracted_text + "\n\n")

        f.write("MATCHED MEDICINES:\n")
        for med in medicines:
            f.write(f"- {med}\n")

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prescription AI Pipeline")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to prescription image")

    args = parser.parse_args()

    run_pipeline(args.image)
"""