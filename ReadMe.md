# Hydr.AI Backend

CURRENT GOOGLE CLOUD LINK
```
https://hydr-ai-backend-529883695650.us-central1.run.app
```
This backend handles sensor data collection, weather integration, and hydration prediction using a pre-trained machine learning model.

---

CURL COMMAND (Replace @data.json with your file)

IF JSON IS A FILE:
```
 curl -X 'POST'   'https://hydr-ai-backend-529883695650.us-central1.run.app/sensor'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d @data.json
```
IF JUST JSON:
```
 curl -X 'POST'   'https://hydr-ai-backend-529883695650.us-central1.run.app/sensor'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "hr": 78, "temp": 36.8, "gsr": 0.5, "lat": 40.7128, "lon": -74.0060
}'
```
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



````markdown
## Updating the Hydr.AI Backend on Google Cloud Run

After making changes to your local `backend` code, follow these steps to update the deployed Cloud Run service:

1. **Test locally (optional but recommended)**  
   From the `backend` folder:
   ```bash
   uvicorn app.main:app --reload
````

Verify everything works locally.

2. **Rebuild the container image**

   ```bash
   gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/hydr-ai-backend
   ```

   Replace `<YOUR_PROJECT_ID>` with your Google Cloud project ID.

3. **Redeploy to the same Cloud Run service**

   ```bash
   gcloud run deploy hydr-ai-backend \
       --image gcr.io/<YOUR_PROJECT_ID>/hydr-ai-backend \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
   ```

   * Keep the service name (`hydr-ai-backend`) the same so your URL does **not** change.

4. **Test the deployed service**
   Use your existing Cloud Run URL:

   ```bash
   curl.exe -X POST "https://hydr-ai-backend-xxxx-uc.a.run.app/sensor" -H "Content-Type: application/json" -d "{}"
   curl.exe "https://hydr-ai-backend-xxxx-uc.a.run.app/recommendation"
   ```

```

