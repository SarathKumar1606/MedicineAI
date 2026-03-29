# =========================================
# IMPORTS
# =========================================
from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

# 🔥 import your existing function
from src.ocr import extract_prescription_text

# =========================================
# CONFIG
# =========================================
UPLOAD_DIR = "D:/Deep Learning Project/prescription-ai/src"   # 🔥 CHANGE THIS PATH

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================
# APP
# =========================================
app = FastAPI()

# =========================================
# API ENDPOINT
# =========================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # 🔥 unique filename (avoid overwrite)
    file_ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"

    save_path = os.path.join(UPLOAD_DIR, filename)

    # =========================================
    # SAVE FILE LOCALLY
    # =========================================
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"✅ Saved file at: {save_path}")

    # =========================================
    # RUN YOUR EXISTING PIPELINE
    # =========================================
    try:
        lines = extract_prescription_text(save_path)
    except Exception as e:
        return {"error": str(e)}

    # =========================================
    # RETURN RESPONSE (KEEP YOUR UI FORMAT)
    # =========================================
    return {
        "prescription": {
            "confidence_score": 0.92,
            "patient": {},
            "doctor": {},
            "diagnosis": "",
            "medications": [{"name": l, "dosage": "-", "frequency": "-", "duration": "-", "instructions": "-"} for l in lines],
            "tests_ordered": [],
            "notes": "Auto-extracted text"
        }
    }