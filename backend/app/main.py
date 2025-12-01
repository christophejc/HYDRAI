from fastapi import FastAPI
from .routes import router
from .database import Base, engine

# Create all tables in the database (if not already created)
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Hydr.AI Backend",
    description="Backend for ESP32 hydration wearable + dummy data",
    version="1.0.0"
)

# Include the API router
app.include_router(router)

# Optional: root endpoint
@app.get("/")
def root():
    return {"message": "Hydr.AI Backend is running. Visit /docs for API docs."}
