import torch
import easyocr
import numpy as np
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/content/experiments/exp4"
IMAGE_PATH = "/content/test_prescription.jpg"

# -------------------------
# Load Models
# -------------------------
print("Loading EasyOCR detector...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

# -------------------------
# Load Image
# -------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

# -------------------------
# Detect text regions
# -------------------------
results = reader.readtext(image_np)

print(f"Detected {len(results)} text regions")

# -------------------------
# Function: Crop box
# -------------------------
def crop_box(image, box):
    x_coords = [int(p[0]) for p in box]
    y_coords = [int(p[1]) for p in box]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return image.crop((x_min, y_min, x_max, y_max))

# -------------------------
# Process each region
# -------------------------
all_texts = []

for i, (bbox, text, conf) in enumerate(results):
    cropped = crop_box(image, bbox)

    # Resize (IMPORTANT)
    cropped = cropped.resize((384, 384))

    pixel_values = processor(
        images=cropped,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=32,
            num_beams=4
        )

    pred_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    all_texts.append(pred_text)

    print(f"\n🔹 Region {i+1}")
    print(f"EasyOCR: {text}")
    print(f"TrOCR  : {pred_text}")

# -------------------------
# Final Output
# -------------------------
print("\n🧾 FINAL EXTRACTED TEXT:")
print(" ".join(all_texts))