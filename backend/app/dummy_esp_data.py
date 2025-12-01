import random

def get_dummy_sensor():
    """Return a single fake sensor reading with coordinates."""
    return {
        "hr": random.randint(60, 120),                  # heart rate
        "temp": round(random.uniform(36.0, 38.0), 1),  # body temperature
        "gsr": round(random.uniform(0.2, 1.0), 2),     # galvanic skin response
        "lat": round(random.uniform(40.5, 40.9), 6),   # example NY lat
        "lon": round(random.uniform(-74.2, -73.7), 6)  # example NY lon
    }

def get_dummy_history(n=10):
    """Return a list of n fake sensor readings."""
    return [get_dummy_sensor() for _ in range(n)]
