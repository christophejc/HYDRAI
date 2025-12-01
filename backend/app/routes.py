from fastapi import APIRouter
from .weather_api import get_weather
from .ml_engine import predict_hydration_level
from .dummy_esp_data import get_dummy_sensor
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import SensorData

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/sensor")
def receive_sensor_data(data: dict, db: Session = Depends(get_db)):
    """
    Receive sensor data including lat/lon from ESP32 or dummy source.
    Expected JSON:
    {
        "hr": 78,
        "temp": 36.8,
        "gsr": 0.5,
        "lat": 40.7128,
        "lon": -74.0060
    }
    """
    if not data:
        # Use dummy sensor data if no JSON is provided
        data = get_dummy_sensor()
        print("Using dummy sensor data:", data)

    entry = SensorData(
        heart_rate=data.get("hr"),
        temperature=data.get("temp"),
        gsr=data.get("gsr"),
        lat=data.get("lat"),
        lon=data.get("lon")
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {"status": "ok", "id": entry.id}

@router.get("/recommendation")
def recommendation(db: Session = Depends(get_db)):
    latest = db.query(SensorData).order_by(SensorData.id.desc()).first()

    if not latest:
        return {"error": "No sensor data available"}

    # Use sensor lat/lon or default NYC
    lat = latest.lat or 40.7
    lon = latest.lon or -74.0

    # Fetch weather
    weather = get_weather(lat=lat, lon=lon)

    humidity = weather.get("main", {}).get("humidity", 50)
    outside_temp = weather.get("main", {}).get("temp", 20)

    # Placeholder — you will replace this with ESP32 steps later
    steps = 5000

    # ML prediction
    result = predict_hydration_level(
        hr=latest.heart_rate,
        temp=latest.temperature,
        gsr=latest.gsr,
        humidity=humidity,
        outside_temp=outside_temp,
        steps=steps,
    )

    return {
        "hydration_prediction": result,
        "weather": {
            "humidity": humidity,
            "outside_temp": outside_temp
        },
        "latest_sensor": {
            "hr": latest.heart_rate,
            "temp": latest.temperature,
            "gsr": latest.gsr,
            "lat": latest.lat,
            "lon": latest.lon
        }
    }
