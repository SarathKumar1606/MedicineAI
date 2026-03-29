# =========================================
# IMPORTS
# =========================================
import easyocr
import torch
import numpy as np
from PIL import Image
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# =========================================
# DEVICE
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
# LOAD MODELS (LOAD ONCE)
# =========================================
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

trocr_model.to(DEVICE)
trocr_model.eval()

# =========================================
# TEXT CLEANING
# =========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================================
# GROUP LINES
# =========================================
def group_lines(results, y_thresh=15):

    results = sorted(results, key=lambda x: min([p[1] for p in x[0]]))

    lines = []
    current = []
    current_y = None

    for bbox, text, _ in results:
        y = min([p[1] for p in bbox])

        if current_y is None or abs(y - current_y) < y_thresh:
            current.append(bbox)
            current_y = y
        else:
            lines.append(current)
            current = [bbox]
            current_y = y

    if current:
        lines.append(current)

    return lines

# =========================================
# TrOCR LINE READER (ONLY SMALL CROPS)
# =========================================
def read_line_trocr(image, boxes):

    xs, ys = [], []

    for b in boxes:
        xs += [int(p[0]) for p in b]
        ys += [int(p[1]) for p in b]

    crop = image.crop((min(xs), min(ys), max(xs), max(ys)))

    # 🔥 Resize small → faster inference
    crop = crop.resize((384, 384))

    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        ids = trocr_model.generate(pixel_values, max_length=32)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    return text

# =========================================
# MAIN PIPELINE
# =========================================
def extract_prescription_text(image_path):

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # STEP 1: FAST detection
    results = reader.readtext(image_np)

    # STEP 2: group lines
    lines = group_lines(results)

    final_lines = []

    for line_boxes in lines:

        # 🔥 Only run TrOCR on likely useful lines
        text = read_line_trocr(image, line_boxes)
        cleaned = clean_text(text)

        if len(cleaned) > 2:
            final_lines.append(cleaned)

    return final_lines

# =========================================
# TEST
# =========================================
if __name__ == "__main__":

    image_path = "/content/test_prescription.jpg"

    lines = extract_prescription_text(image_path)

    print("\n🧾 Final Clean Prescription:\n")

    for l in lines:
        print("-", l)