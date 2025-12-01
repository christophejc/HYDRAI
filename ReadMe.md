# Hydr.AI Backend

This backend handles sensor data collection, weather integration, and hydration prediction using a pre-trained machine learning model.

---

## Requirements

- Python 3.11+
- Pip packages: `fastapi`, `uvicorn`, `sqlalchemy`, `joblib`, `requests`, `pydantic`, `numpy`
- SQLite (default database)

---

s

## Quick start — step by step

1. **Clone the repository** and change into the backend folder:
```powershell
git clone <your-repo-url>
cd backend
```

2. **Install dependencies:**
```powershell
pip install -r requirements.txt
```
3. **Train the model to produce hydration_classifier.pkl (run the training script from the backend folder):**
```powershell
python app/train_model.py
```
4. **Start the backend server (Terminal 1 — keep this running):**
```powershell
cd backend
uvicorn app.main:app --reload
```
Server will run at: http://127.0.0.1:8000

5. **Insert dummy sensor data (Terminal 2 — PowerShell):**
```powershell
curl.exe -X POST http://127.0.0.1:8000/sensor -H "Content-Type: application/json" -d "{}"
```
This will send dummy data

6. **Get a hydration recommendation (Terminal 2):**
```powershell
curl.exe http://127.0.0.1:8000/recommendation
```
