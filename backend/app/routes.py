from fastapi import APIRouter, Depends
from .weather_api import get_weather
from .ml_engine import predict_hydration_level
from .dummy_esp_data import get_dummy_sensor, get_dummy_apple
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import SensorData, AppleHealthData
from .preprocessor import calculate_accurate_bpm, calculate_temp_c_from_raw, calculate_gsr_resistance_from_raw

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
    Expected RAW data format from ESP:
    {
        "ir_readings": [...],
        "raw_temp": 20000,
        "raw_gsr": 3000,
        "sampling_rate": 50,
        "lat": 40.7128,
        "lon": -74.0060
    }
    """
    if not data:
        # Use dummy sensor data if no JSON is provided
        data = get_dummy_sensor()
        print("Using dummy sensor data:", data)

    ir = data.get("ir_readings", [])
    raw_temp = data.get("raw_temp", 0)
    raw_gsr = data.get("raw_gsr", 0)
    fs = data.get("sampling_rate", 50)
    lat = data.get("lat")
    lon = data.get("lon")


    bpm = calculate_accurate_bpm(ir, fs)
    temp_c = calculate_temp_c_from_raw(raw_temp)
    gsr_ohms = calculate_gsr_resistance_from_raw(raw_gsr)

    print("\n[SENSOR DATA RECEIVED]")
    print(f"  IR samples: {len(ir)}")
    print(f"  Raw Temp: {raw_temp}")
    print(f"  Raw GSR:  {raw_gsr}")
    print(f"  Computed: BPM={bpm}, Temp={temp_c}, GSR={gsr_ohms}Ω")
    
    # --- Store in database ---
    entry = SensorData(
        heart_rate=bpm,
        temperature=temp_c,
        gsr=gsr_ohms,
        lat=lat,
        lon=lon
    )
    
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {"status_esp": "ok", "id": entry.id}

# ---------- NEW APPLE HEALTH ROUTE ----------
@router.post("/apple-health")
def receive_apple_health(data: dict, db: Session = Depends(get_db)):
    """
    Expected JSON format:
    {
      "data": {
        "metrics": [
          {
            "name": "step_count",
            "units": "count",
            "data": [{
              "qty": 4033,
              "date": "2025-12-03 00:00:00 -0500"
            }]
          },
          {
            "name": "active_energy",
            "units": "kcal",
            "data": [{
              "qty": 314.38,
              "date": "2025-12-03 00:00:00 -0500"
            }]
          }
        ]
      }
    }
    """
    if not data:
        # Use dummy sensor data if no JSON is provided
        data = get_dummy_apple()
        print("Using dummy sensor data:", data)

    metrics = data.get("data", {}).get("metrics", [])

    steps = None
    energy = None

    for metric in metrics:
        name = metric.get("name")
        entries = metric.get("data", [])

        if not entries:
            continue

        first_entry = entries[0]

        if name == "step_count":
            steps = first_entry.get("qty")

        elif name == "active_energy":
            energy = first_entry.get("qty")

    if steps is None and energy is None:
        return {"error": "No valid Apple Health data found"}

    entry = AppleHealthData(
        step_count=steps,
        active_energy=energy,
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "status": "ok",
        "id": entry.id,
        "steps": steps,
        "active_energy": energy,
    }


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
