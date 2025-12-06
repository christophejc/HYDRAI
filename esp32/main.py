from machine import Pin, I2C
from time import sleep_ms
from max30102 import MAX30102
import network
import urequests
import ujson
from ads1x15 import ADS1115

# --- CONFIGURATION ---
WIFI_SSID = "XPS"
WIFI_PASS = "12345678"
SERVER_IP = "10.206.222.136"
SERVER_PORT = "5000"

# Note: We now send everything to ONE endpoint
URL_PROCESS = f"http://{SERVER_IP}:5000/process_data"

# Pins
I2C_SCL_PIN = 20  
I2C_SDA_PIN = 22

# Global Location Variables
DEVICE_LAT = 0.0
DEVICE_LON = 0.0

def get_geolocation():
    """
    Fetches approximate location based on IP address.
    """
    print("Fetching Geolocation...")
    try:
        # ip-api.com is free for non-commercial use (HTTP only)
        res = urequests.get('http://ip-api.com/json')
        if res.status_code == 200:
            data = res.json()
            lat = data.get('lat', 0.0)
            lon = data.get('lon', 0.0)
            print(f"Location Found: {lat}, {lon}")
            res.close()
            return lat, lon
        res.close()
    except Exception as e:
        print(f"Geo-Location Failed: {e}")
    
    return 0.0, 0.0

def main():
    global DEVICE_LAT, DEVICE_LON
    print("Initializing Raw Data Collector...")

    # 1. WiFi Connection
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print(f"Connecting to {WIFI_SSID}...")
        wlan.connect(WIFI_SSID, WIFI_PASS)
        while not wlan.isconnected():
            sleep_ms(500)
            print(".", end="")
    print(f"\nWiFi Connected! IP: {wlan.ifconfig()[0]}")

    # 2. Get Location (Do this once to save time in the loop)
    DEVICE_LAT, DEVICE_LON = get_geolocation()

    # 3. I2C Setup
    try:
        i2c = I2C(0, scl=Pin(I2C_SCL_PIN), sda=Pin(I2C_SDA_PIN), freq=400000)
    except Exception as e:
        print(f"I2C Init Error: {e}")
        return

    # 4. Sensor Setup
    # ADC (ADS1115)
    try:
        adc = ADS1115(i2c, address=0x48)
        print("ADS1115 found.")
    except Exception as e:
        print(f"ADS1115 Error: {e}")
        return

    # Pulse Ox (MAX30102)
    try:
        sensor = MAX30102(i2c=i2c)
        sensor.setup_sensor()
        sensor.set_sample_rate(400)
        sensor.set_fifo_average(8)
        sensor.set_active_leds_amplitude(255)
        print("MAX30102 Configured.")
    except Exception as e:
        print(f"MAX30102 Error: {e}")
        return

    # Buffer Config
    SAMPLE_RATE = 50 
    SECONDS_TO_LOG = 10
    BUFFER_SIZE = SAMPLE_RATE * SECONDS_TO_LOG 
    ir_buffer = []

    print("Starting Loop. Gathering data...")

    while True:
        sensor.check()

        if sensor.available():
            sensor.pop_red_from_storage()
            ir_reading = sensor.pop_ir_from_storage()
            
            ir_buffer.append(ir_reading)

            # --- BUFFER FULL LOGIC ---
            if len(ir_buffer) >= BUFFER_SIZE:
                print(f"\nBuffer Full ({len(ir_buffer)}). Reading Analog & Uploading...")
                
                try:
                    # 1. Read RAW Analog Sensors (No math here!)
                    raw_temp = adc.read_channel(0)
                    raw_gsr = adc.read_channel(1)
                    print(f"   Raw Values -> Temp: {raw_temp}, GSR: {raw_gsr}")

                    # 2. Construct Payload (Now includes Location)
                    payload = {
                        "sampling_rate": SAMPLE_RATE,
                        "ir_readings": ir_buffer,
                        "raw_temp": raw_temp,
                        "raw_gsr": raw_gsr,
                        "lat": DEVICE_LAT,
                        "lon": DEVICE_LON
                    }
                    print(payload)
                    # 3. Send to Server
                    headers = {'Content-Type': 'application/json'}
                    res = urequests.post(URL_PROCESS, data=ujson.dumps(payload), headers=headers)
                    print(f"Raw Response: {res.text}")
                    if res.status_code == 200:
                        data = res.json()
                        print("   Server Response:")
                        print(f"     HR:   {data.get('hr')} BPM")
                        print(f"     Temp: {data.get('temp')} C")
                        print(f"     GSR:  {data.get('gsr')} Ohms")
                    else:
                        print(f"   Server Error: {res.status_code}")
                    
                    res.close()
                    
                except Exception as e:
                    print(f"   Upload Failed: {e}")

                # Reset Buffer
                ir_buffer = []

if __name__ == "__main__":
    main()