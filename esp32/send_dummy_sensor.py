import urequests
import time
import random

# ------------------------------
# CONFIG
# ------------------------------
SERVER_URL = "http://<your-pc-ip>:8000/sensor"  # replace with your PC/backend IP
USE_DUMMY = True  # Set to False when using real sensors
SEND_INTERVAL = 5  # seconds between sends

# ------------------------------
# SENSOR FUNCTIONS
# ------------------------------
def get_dummy_sensor():
    """Return fake sensor reading with coordinates."""
    return {
        "hr": random.randint(60, 120),
        "temp": round(random.uniform(36.0, 38.0), 1),
        "gsr": round(random.uniform(0.2, 1.0), 2),
        "lat": round(random.uniform(40.5, 40.9), 6),   # example NY area
        "lon": round(random.uniform(-74.2, -73.7), 6)
    }

def get_real_sensor():
    """
    Read actual sensors connected to the ESP32.
    Replace these placeholders with your sensor code.
    """
    hr = 80         # replace with actual heart rate sensor reading
    temp = 37.0     # replace with actual temperature sensor reading
    gsr = 0.5       # replace with actual GSR reading
    lat = 40.7128   # replace with actual GPS reading
    lon = -74.0060  # replace with actual GPS reading

    return {
        "hr": hr,
        "temp": temp,
        "gsr": gsr,
        "lat": lat,
        "lon": lon
    }

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    if USE_DUMMY:
        data = get_dummy_sensor()
    else:
        data = get_real_sensor()

    try:
        response = urequests.post(SERVER_URL, json=data)
        print("Sent data:", data)
        print("Response:", response.text)
    except Exception as e:
        print("Error sending data:", e)

    time.sleep(SEND_INTERVAL)
