import json
import numpy as np
from scipy.signal import find_peaks
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import requests # pip install requests

# --- CONFIGURATION ---
HOST_NAME = "0.0.0.0"  
PORT_NUMBER = 5000     

# External Cloud Endpoint
EXTERNAL_CLOUD_URL = "https://hydr-ai-backend-529883695650.us-central1.run.app/sensor"

# --- CONSTANTS FOR CALCULATIONS ---
VOLTS_PER_BIT = 4.096 / 32767.0
V_SOURCE = 3.3      
R_FIXED = 1000.0   # 1k Resistor
SERIAL_CALIBRATION = 12324 

# Standard 10K 3950 NTC Lookup Table
THERMISTOR_TABLE = [
    (175200, -30), (97070, -20), (55330, -10), 
    (32650, 0),  (25390, 5),  (19900, 10), (15710, 15),
    (12490, 20), (10000, 25), (8057, 30),  (6531, 35),
    (5327, 40),  (4369, 45),  (3603, 50),  (2986, 55),
    (2488, 60),  (2083, 65),  (1752, 70),  (1481, 75),
    (1258, 80),  (1072, 85),  (918, 90),   (789, 95),
    (680, 100)
]

def raw_to_volts(raw):
    return raw * VOLTS_PER_BIT

def calculate_temp_c_from_raw(raw_val):
    voltage = raw_to_volts(raw_val)
    if voltage >= V_SOURCE - 0.01: return 0.0 
    if voltage <= 0.01: return 0.0 

    try:
        r_ntc = (voltage * R_FIXED) / (V_SOURCE - voltage)
    except Exception:
        return 0.0
    
    if r_ntc >= THERMISTOR_TABLE[0][0]: return THERMISTOR_TABLE[0][1]
    if r_ntc <= THERMISTOR_TABLE[-1][0]: return THERMISTOR_TABLE[-1][1]

    for i in range(len(THERMISTOR_TABLE) - 1):
        r_high = THERMISTOR_TABLE[i][0]
        r_low  = THERMISTOR_TABLE[i+1][0]
        if r_ntc <= r_high and r_ntc > r_low:
            t_high = THERMISTOR_TABLE[i][1]
            t_low  = THERMISTOR_TABLE[i+1][1]
            fraction = (r_high - r_ntc) / (r_high - r_low)
            temp_c = t_high + (fraction * (t_low - t_high))
            return round(temp_c, 2)
    return 0.0

def calculate_gsr_resistance_from_raw(raw_val):
    if raw_val >= SERIAL_CALIBRATION: return 0 
    try:
        numerator = (32768 + 2 * raw_val) * 10000
        denominator = SERIAL_CALIBRATION - raw_val
        return int(numerator / denominator)
    except Exception:
        return 0

def calculate_accurate_bpm(raw_data, fs):
    if len(raw_data) < 50: return 0, 0
    sig = np.array(raw_data)
    sig = sig - np.mean(sig) 
    min_dist = int(fs * 0.3) 
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=50)
    if len(peaks) < 2: return 0, len(peaks)
    avg_distance = np.mean(np.diff(peaks))
    bpm = 60.0 / (avg_distance / fs)
    return bpm, len(peaks)

def forward_to_cloud(payload):
    """
    Sends the processed JSON to the external Google Cloud Run endpoint.
    """
    try:
        print(f"   [CLOUD] Forwarding data to {EXTERNAL_CLOUD_URL}...")
        
        # Headers specifically for your curl request
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Requests.post automatically handles JSON serialization
        response = requests.post(EXTERNAL_CLOUD_URL, json=payload, headers=headers)
        
        print(f"   [CLOUD] Status: {response.status_code}, Response: {response.text}")
        
    except Exception as e:
        print(f"   [CLOUD] Upload Failed: {e}")

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
                
                # Calculations
                bpm, beats = calculate_accurate_bpm(ir_readings, fs)
                temp_c = calculate_temp_c_from_raw(raw_temp)
                gsr_ohms = calculate_gsr_resistance_from_raw(raw_gsr)
                
                print(f"   Input: {len(ir_readings)} IR samples, Raw Temp: {raw_temp}, Raw GSR: {raw_gsr}")
                print(f"   Calculated -> BPM: {bpm:.1f}, Temp: {temp_c}C, GSR: {gsr_ohms} Ohms")
                
                # 1. Prepare Final JSON
                # Matching the keys from your image: 'hr', 'temp', 'gsr', 'lat', 'lon'
                final_packet = {
                    "hr": round(bpm, 2),
                    "temp": temp_c,
                    "gsr": gsr_ohms, 
                    "lat": lat, 
                    "lon": lon
                }
                
                # 2. Forward to External Cloud
                forward_to_cloud(json_data)
                
                # 3. Respond to ESP32 (Success)
                # We send the same packet back so ESP32 knows what was calculated
                final_packet["status"] = "success"
                self.send_json(200, final_packet)

            else:
                self.send_json(404, {"error": "Use /process_data endpoint"})

        except Exception as e:
            print(f"Error: {e}")
            self.send_json(500, {"error": str(e)})

    def send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
        
    def log_message(self, format, *args):
        return 

if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server started on {HOST_NAME}:{PORT_NUMBER}")
    print(f"Endpoint: http://{local_ip}:{PORT_NUMBER}/process_data")
    print(f"Forwarding to: {EXTERNAL_CLOUD_URL}")
    
    server = HTTPServer((HOST_NAME, PORT_NUMBER), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()