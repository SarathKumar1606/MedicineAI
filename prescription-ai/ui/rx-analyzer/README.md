# 💊 RxAnalyzer — Prescription Intelligence

A full-stack application for DL-based prescription analysis.
Upload a `.png` / `.jpg` prescription image → FastAPI backend → structured JSON → beautiful UI output.

---

## 📁 Project Structure

```
rx-analyzer/
├── backend/
│   ├── main.py               ← FastAPI app (entry point)
│   ├── requirements.txt      ← Python dependencies
│   ├── uploads/              ← Uploaded images stored here (auto-created)
│   └── models/
│       └── dl_model.py       ← DROP YOUR DL MODEL HERE
│
├── frontend/
│   └── index.html            ← Complete frontend (zero dependencies)
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be live at → `http://127.0.0.1:8000`  
Swagger docs → `http://127.0.0.1:8000/docs`

### 2. Frontend

Just open `frontend/index.html` in your browser.  
No build step. No npm install. Pure HTML/CSS/JS.

---

## 🔌 Connecting Your DL Model

1. Open `backend/models/dl_model.py`
2. Implement `run_inference(image_path: str) -> dict`
3. In `backend/main.py`, replace the mock call:

```python
# Replace this line:
result = mock_analyze(str(save_path), file.filename)

# With your model:
from models.dl_model import run_inference
result = run_inference(str(save_path))
```

### Expected JSON output schema

```json
{
  "status": "success",
  "file": "prescription.png",
  "prescription": {
    "patient":      { "name": "", "age": "", "gender": "", "date": "" },
    "doctor":       { "name": "", "registration_no": "", "hospital": "", "contact": "" },
    "diagnosis":    "...",
    "medications":  [
      {
        "name": "", "dosage": "", "frequency": "",
        "duration": "", "instructions": ""
      }
    ],
    "tests_ordered":    ["..."],
    "follow_up":        "...",
    "notes":            "...",
    "confidence_score": 0.95
  }
}
```

---

## 🛠️ API Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Health check                         |
| POST   | `/analyze` | Upload image → returns JSON          |

**POST /analyze** — Form field: `file` (multipart/form-data, image/png or image/jpeg)

---

## ⚙️ Frontend Config

Edit the first line of the `<script>` block in `frontend/index.html`:

```js
const API_URL = "http://127.0.0.1:8000/analyze";
// Change this if your backend runs on a different host/port
```

---

## 🏗️ Tech Stack

| Layer    | Technology            |
|----------|-----------------------|
| Frontend | Vanilla HTML/CSS/JS   |
| Backend  | Python + FastAPI      |
| Model    | Your DL model (plug-in) |
| Bridge   | REST API (JSON)       |
