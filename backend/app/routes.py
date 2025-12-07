import pandas as pd
from fastapi import APIRouter, Depends
from .weather_api import get_weather
from .inference import classify_window_single
from .dummy_esp_data import get_dummy_sensor, get_dummy_apple
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import SensorData, AppleHealthData
from .preprocessor import calculate_accurate_bpm, calculate_temp_c_from_raw, calculate_gsr_resistance_from_raw, interpolate_sensor_data
from .llm_generator import generate_llm_response
from .notifications import send_notification

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
async def recommendation(db: Session = Depends(get_db)):
    
    # Fetch all sensor data
    sensor_rows = db.query(SensorData).all()

    if not sensor_rows:
        return {"error": "No SensorData available."}

    data_list = [
        {
            # Note: We use the *interpolated* column names from your original script
            'timestamp': row.timestamp,
            'gsr': row.gsr,
            'temperature': row.temperature,
            'heart_rate': row.heart_rate,
            # Add steps/active_calories if they were part of SensorData, 
            # otherwise they'll be NaN and dropped later or come from AppleHealthData
            # For simplicity, we assume the DB has the 3 main metrics + timestamp:
        }
        for row in sensor_rows
    ]
    sensor_df = pd.DataFrame(data_list)
    
    # Ensure timestamp is a datetime object before processing
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])

    # Apply the interpolation and resampling logic
    # This generates a high-frequency, clean DataFrame for accurate averaging
    processed_df = interpolate_sensor_data(sensor_df)  

    #print("Processed DataFrame head:\n", processed_df.head())

    selection_cols = ['gsr', 'temperature', 'heart_rate']

    # 2. Randomly sample 150 rows
    # We use random_state=42 for reproducibility of the random sample.
    df_sampled = processed_df.head(n=150)

    # 3. Select the desired columns and convert to a NumPy array
    np_array = df_sampled[selection_cols].to_numpy()

    # The final np_array has the shape (150, 3)
    #print("Sampled NumPy array shape:", np_array.shape)
    #print(np_array)

    label = classify_window_single(np_array)
    print("Predicted Hydration Level:", label)

    

    latest_health = (
        db.query(AppleHealthData)
        .order_by(AppleHealthData.id.desc())
        .first()
    )

    if latest_health:
        calories = latest_health.active_energy
        step_count = latest_health.step_count
    else:
        calories = 0
        step_count = 0

    print("Latest Apple Health - Calories:", calories, "Steps:", step_count)

    
    latest = db.query(SensorData).order_by(SensorData.id.desc()).first()
    if not latest:
        return {"error": "No sensor data available"}

    lat = latest.lat or 40.7
    lon = latest.lon or -74.0

    # Fetch weather
    weather = get_weather(lat=lat, lon=lon)

    # 3. Generate LLM Personalized Response
    '''llm_advice = generate_llm_response(
        label,
        step_count,
        calories,
        weather
    )'''
    llm_advice = await generate_llm_response(label, step_count, calories, weather)
    
    # 4. Send Notification via ntfy.sh
    
    # Determine priority based on risk
    priority = 5 if label == "dehydrated" else 3 
    tags = "warning,alert" if label == "dehydrated" else "info,hydration"

    send_notification(
        title=f"HYDR-AI Alert: {label.upper()}",
        message=llm_advice,
        priority=priority,
        tags=tags
    )

    # 5. Return LLM Advice in the API Response
    return {
        "hydration_prediction": label,
        "personalized_advice": llm_advice,
    }