# =========================================
# IMPORTS
# =========================================
import easyocr
import torch
import numpy as np
from PIL import Image
import re
import json

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModel
)

from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# DEVICE
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
# LOAD OCR MODELS
# =========================================
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

trocr_model.to(DEVICE)
trocr_model.eval()

# =========================================
# LOAD BIOBERT
# =========================================
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

bio_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
bio_model = AutoModel.from_pretrained(BIOBERT_MODEL)

bio_model.to(DEVICE)
bio_model.eval()

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
# TrOCR LINE READER
# =========================================
def read_line_trocr(image, boxes):

    xs, ys = [], []

    for b in boxes:
        xs += [int(p[0]) for p in b]
        ys += [int(p[1]) for p in b]

    crop = image.crop((min(xs), min(ys), max(xs), max(ys)))
    crop = crop.resize((384, 384))

    pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        ids = trocr_model.generate(pixel_values, max_length=32)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    return text

# =========================================
# BIOBERT EMBEDDING
# =========================================
def get_embedding(text):

    inputs = bio_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bio_model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

    return embedding.squeeze().cpu().numpy()

# =========================================
# LOAD MEDICINE DATABASE
# =========================================
def load_medicine_db(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    names = [item["name"] for item in data]

    return names

# =========================================
# BUILD EMBEDDING DATABASE
# =========================================
def build_embedding_db(medicine_names):

    embeddings = []

    print("Building medicine embeddings...")

    for i, name in enumerate(medicine_names):
        emb = get_embedding(name)
        embeddings.append(emb)

        if i % 200 == 0:
            print(f"Processed {i}/{len(medicine_names)}")

    return np.array(embeddings)

# =========================================
# TOP-K MATCHING
# =========================================
def get_top_k_matches(query, med_names, med_embeddings, k=5):

    query_emb = get_embedding(query)

    sims = cosine_similarity([query_emb], med_embeddings)[0]

    top_k_idx = sims.argsort()[-k:][::-1]

    results = [(med_names[i], sims[i]) for i in top_k_idx]

    return results

# =========================================
# MAIN OCR PIPELINE
# =========================================
def extract_prescription_text(image_path):

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    results = reader.readtext(image_np)
    lines = group_lines(results)

    final_lines = []

    for line_boxes in lines:

        text = read_line_trocr(image, line_boxes)
        cleaned = clean_text(text)

        if len(cleaned) > 2:
            final_lines.append(cleaned)

    return final_lines

# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":

    image_path = "test_prescription.jpg"
    medicine_json = "medicines_3000.json"

    # Load medicine DB
    medicine_names = load_medicine_db(medicine_json)

    # Build embeddings (run once → then save for reuse)
    medicine_embeddings = build_embedding_db(medicine_names)

    print("\n🧾 OCR + BioBERT Matching:\n")

    # OCR extraction
    lines = extract_prescription_text(image_path)

    # Match each line
    for l in lines:

        print(f"\n🔹 OCR: {l}")

        matches = get_top_k_matches(l, medicine_names, medicine_embeddings, k=5)

        for name, score in matches:
            print(f"   → {name} ({score:.3f})")