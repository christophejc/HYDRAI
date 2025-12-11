import json
import numpy as np
from scipy.signal import find_peaks
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import requests
import csv
import os
from datetime import datetime
import threading
import time

# --- CONFIGURATION ---
HOST_NAME = "0.0.0.0"
PORT_NUMBER = 5000
CSV_FILENAME = "sensor_log.csv"

# External Cloud Endpoints
EXTERNAL_CLOUD_URL = "https://hydr-ai-backend-529883695650.us-central1.run.app/sensor"
RECOMMENDATION_CLOUD_URL = "https://hydr-ai-backend-529883695650.us-central1.run.app/recommendation"
SCHEDULE_INTERVAL_SECONDS = 2 * 60  # 2 minutes

# --- SCHEDULER FUNCTIONS ---

def get_recommendation_from_cloud(url):
    """Trigger GET request to cloud recommendation endpoint."""
    try:
        print(f"\n[SCHEDULER] Requesting recommendation from {url}...")
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            try:
                result = response.json()
                advice = result.get('personalized_advice', 'N/A')
                print(
                    f"  [SCHEDULER] Success (200). "
                    f"Prediction: {result.get('hydration_prediction')}, "
                    f"Advice: {advice[:50]}..."
                )
            except json.JSONDecodeError:
                print(f"  [SCHEDULER] Success but invalid JSON: {response.text[:100]}...")
        else:
            print(f"  [SCHEDULER] Failed ({response.status_code}) → {response.text}")

    except Exception as e:
        print(f"  [SCHEDULER] Request Failed: {e}")


class RecommendationScheduler(threading.Thread):
    def __init__(self, interval, url):
        super().__init__()
        self.interval = interval
        self.url = url
        self.running = True
        self.daemon = True  # allows program exit even if running

    def run(self):
        get_recommendation_from_cloud(self.url)
        while self.running:
            time.sleep(self.interval)
            if not self.running:
                break
            get_recommendation_from_cloud(self.url)

    def stop(self):
        self.running = False


# --- CONSTANTS FOR CALCULATIONS ---
VOLTS_PER_BIT = 4.096 / 32767.0
V_SOURCE = 3.3
R_FIXED = 1000.0
SERIAL_CALIBRATION = 12324

# Standard 10K 3950 NTC Lookup Table
THERMISTOR_TABLE = [
    (175200, -30), (97070, -20), (55330, -10),
    (32650, 0), (25390, 5), (19900, 10), (15710, 15),
    (12490, 20), (10000, 25), (8057, 30), (6531, 35),
    (5327, 40), (4369, 45), (3603, 50), (2986, 55),
    (2488, 60), (2083, 65), (1752, 70), (1481, 75),
    (1258, 80), (1072, 85), (918, 90), (789, 95),
    (680, 100)
]


def raw_to_volts(raw):
    return raw * VOLTS_PER_BIT


def calculate_temp_c_from_raw(raw_val):
    voltage = raw_to_volts(raw_val)

    if voltage >= V_SOURCE - 0.01:
        return 0.0
    if voltage <= 0.01:
        return 0.0

    try:
        r_ntc = (voltage * R_FIXED) / (V_SOURCE - voltage)
    except Exception:
        return 0.0

    if r_ntc >= THERMISTOR_TABLE[0][0]:
        return THERMISTOR_TABLE[0][1]
    if r_ntc <= THERMISTOR_TABLE[-1][0]:
        return THERMISTOR_TABLE[-1][1]

    for i in range(len(THERMISTOR_TABLE) - 1):
        r_high, t_high = THERMISTOR_TABLE[i]
        r_low, t_low = THERMISTOR_TABLE[i + 1]

        if r_low < r_ntc <= r_high:
            fraction = (r_high - r_ntc) / (r_high - r_low)
            temp_c = t_high + fraction * (t_low - t_high)
            return round(temp_c, 2)

    return 0.0


def calculate_gsr_resistance_from_raw(raw_val):
    if raw_val >= SERIAL_CALIBRATION:
        return 0
    try:
        numerator = (32768 + 2 * raw_val) * 10000
        denominator = SERIAL_CALIBRATION - raw_val
        return int(numerator / denominator)
    except Exception:
        return 0


def calculate_accurate_bpm(raw_data, fs):
    if len(raw_data) < 50:
        return 0, 0

    sig = np.array(raw_data)
    sig = sig - np.mean(sig)

    peaks, _ = find_peaks(sig, distance=int(fs * 0.3), prominence=50)

    if len(peaks) < 2:
        return 0, len(peaks)

    avg_dist = np.mean(np.diff(peaks))
    bpm = 60.0 / (avg_dist / fs)
    return bpm, len(peaks)


# --- CSV LOGGING ---
def log_to_csv(data_dict):
    try:
        file_exists = os.path.isfile(CSV_FILENAME)

        with open(CSV_FILENAME, mode='a', newline='') as file:
            fieldnames = [
                'timestamp', 'gsr_raw', 'temp_raw',
                'hr_raw', 'steps', 'active_calories', 'label'
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(data_dict)
            print(f"   [CSV] Row added to {CSV_FILENAME}")

    except Exception as e:
        print(f"   [CSV] Error saving to file: {e}")


def forward_to_cloud(payload):
    try:
        print(f"   [CLOUD] Forwarding to {EXTERNAL_CLOUD_URL}...")
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        response = requests.post(EXTERNAL_CLOUD_URL, json=payload, headers=headers)
        print(f"   [CLOUD] Status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   [CLOUD] Upload Failed: {e}")


# --- HTTP SERVER HANDLER ---
class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            json_data = json.loads(post_data.decode('utf-8'))

            if self.path == '/process_data':
                print("\n[DATA PACKET RECEIVED]")

                ir_readings = json_data.get('ir_readings', [])
                raw_temp = json_data.get('raw_temp', 0)
                raw_gsr = json_data.get('raw_gsr', 0)
                fs = json_data.get('sampling_rate', 50)
                lat = json_data.get('lat', 0.0)
                lon = json_data.get('lon', 0.0)

                bpm, beats = calculate_accurate_bpm(ir_readings, fs)
                temp_c = calculate_temp_c_from_raw(raw_temp)
                gsr_ohms = calculate_gsr_resistance_from_raw(raw_gsr)

                print(f"   Input: {len(ir_readings)} IR samples, Temp Raw={raw_temp}, GSR Raw={raw_gsr}")
                print(f"   Output → BPM={bpm:.1f}, Temp={temp_c}C, GSR={gsr_ohms}Ω")

                # 1. Forward original JSON
                forward_to_cloud(json_data)

                # 2. CSV payload (currently disabled)
                csv_payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "gsr_raw": gsr_ohms,
                    "temp_raw": temp_c,
                    "hr_raw": round(bpm, 2),
                    "steps": 39,
                    "active_calories": 0,
                    "label": "dehydrated-morning"
                }

                # log_to_csv(csv_payload)

                # 3. Response to ESP32
                final_packet = {
                    "hr": round(bpm, 2),
                    "temp": temp_c,
                    "gsr": gsr_ohms,
                    "lat": lat,
                    "lon": lon,
                    "status": "success"
                }
                self.send_json(200, final_packet)

            else:
                self.send_json(404, {"error": "Use /process_data endpoint"})

        except Exception as e:
            print(f"Error: {e}")
            self.send_json(500, {"error": str(e)})

    def send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        return


# --- MAIN ---
if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"Server started on {local_ip}:{PORT_NUMBER}")
    print(f"Logging to: {os.path.abspath(CSV_FILENAME)}")

    scheduler = RecommendationScheduler(SCHEDULE_INTERVAL_SECONDS, RECOMMENDATION_CLOUD_URL)
    scheduler.start()

    server = HTTPServer((HOST_NAME, PORT_NUMBER), RequestHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        scheduler.stop()
        server.server_close()
        print("\nServer and scheduler shut down.")
