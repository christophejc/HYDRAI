import random
import math
from datetime import datetime

def generate_dummy_ir_wave(fs=50, duration_sec=2, bpm=75):
    """
    Generates synthetic IR waveform data with clear heart-rate peaks.
    """
    num_samples = fs * duration_sec
    ir_values = []

    # Convert BPM → frequency (Hz)
    freq = bpm / 60.0  # beats per second

    base = 1000  # DC offset

    for n in range(num_samples):
        # Sine wave simulating heartbeat
        t = n / fs
        heartbeat = 80 * math.sin(2 * math.pi * freq * t)  # amplitude 80

        # Add noise
        noise = random.uniform(-10, 10)

        value = int(base + heartbeat + noise)
        ir_values.append(value)

    return ir_values

def get_dummy_sensor():
    """Return a fake RAW sensor payload matching the new expected JSON."""
    
    # Generate fake IR samples for heart rate calculation
    ir_values = generate_dummy_ir_wave(
        fs=50,
        duration_sec=2,   # 2 seconds → 100 samples
        bpm=75            # synthetic heart rate
    )

    return {
        "raw_temp": random.randint(470, 530),   # raw ADC-like value
        "raw_gsr": random.randint(500, 3000),   # raw ADC GSR input
        "ir_readings": ir_values,                 # raw optical sensor signal
        "sampling_rate": 50,
        "lat": round(random.uniform(40.5, 40.9), 6),
        "lon": round(random.uniform(-74.2, -73.7), 6)
    }

def get_dummy_history(n=10):
    """Return a list of n fake sensor readings."""
    return [get_dummy_sensor() for _ in range(n)]

def get_dummy_apple():
    return {
        "data": {
            "metrics": [
                {
                    "name": "step_count",
                    "units": "count",
                    "data": [
                        {
                            "source": "Dummy Watch",
                            "date": str(datetime.now()),
                            "qty": random.randint(2000, 8000)
                        }
                    ]
                },
                {
                    "name": "active_energy",
                    "units": "kcal",
                    "data": [
                        {
                            "source": "Dummy Watch",
                            "date": str(datetime.now()),
                            "qty": round(random.uniform(250.0, 600.0), 2)
                        }
                    ]
                }
            ]
        }
    }