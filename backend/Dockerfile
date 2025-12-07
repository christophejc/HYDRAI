FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN gsutil cp gs://hydr-ai-ml-assets/checkpoints/normwear_last_checkpoint-15470-correct.pth .

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]