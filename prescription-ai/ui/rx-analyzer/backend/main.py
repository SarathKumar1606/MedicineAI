from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import json
from pathlib import Path

app = FastAPI(title="RxAnalyzer API", version="1.0.0")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "RxAnalyzer API is running", "version": "1.0.0"}


@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...)):
    """
    Accepts a prescription image (.png / .jpg) and returns extracted data as JSON.
    Replace the mock_analyze() call below with your actual DL model inference.
    """
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PNG/JPG images are accepted."
        )

    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{file_id}{file_extension}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---------------------------------------------------------------
    # TODO: Replace mock_analyze() with your actual DL model call
    # Example:
    #   from models.dl_model import run_inference
    #   result = run_inference(str(save_path))
    # ---------------------------------------------------------------
    result = mock_analyze(str(save_path), file.filename)

    return JSONResponse(content=result)


def mock_analyze(image_path: str, filename: str) -> dict:
    """
    Mock function — replace with your real DL model inference.
    Returns a structured prescription JSON.
    """
    return {
        "status": "success",
        "file": filename,
        "prescription": {
            "patient": {
                "name": "John Doe",
                "age": "45",
                "gender": "Male",
                "date": "2026-03-25"
            },
            "doctor": {
                "name": "Dr. S. Ramalingam",
                "registration_no": "TN-MCI-45892",
                "hospital": "Apollo Speciality Hospital",
                "contact": "+91 98765 43210"
            },
            "diagnosis": "Hypertension, Type 2 Diabetes",
            "medications": [
                {
                    "name": "Metformin",
                    "dosage": "500 mg",
                    "frequency": "Twice daily",
                    "duration": "30 days",
                    "instructions": "Take after meals"
                },
                {
                    "name": "Amlodipine",
                    "dosage": "5 mg",
                    "frequency": "Once daily",
                    "duration": "30 days",
                    "instructions": "Take in the morning"
                },
                {
                    "name": "Atorvastatin",
                    "dosage": "10 mg",
                    "frequency": "Once daily",
                    "duration": "30 days",
                    "instructions": "Take at bedtime"
                }
            ],
            "tests_ordered": [
                "Fasting Blood Sugar",
                "HbA1c",
                "Lipid Profile"
            ],
            "follow_up": "After 4 weeks",
            "notes": "Avoid high-sodium foods. Exercise 30 min daily. Monitor BP at home.",
            "confidence_score": 0.94
        }
    }
