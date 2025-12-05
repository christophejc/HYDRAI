import numpy as np
from scipy.signal import find_peaks

VOLTS_PER_BIT = 4.096 / 32767.0
V_SOURCE = 3.3
R_FIXED = 1000.0   # 1k resistor
SERIAL_CALIBRATION = 12324

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
    if voltage >= V_SOURCE - 0.01: return 0.0
    if voltage <= 0.01: return 0.0

    try:
        r_ntc = (voltage * R_FIXED) / (V_SOURCE - voltage)
    except:
        return 0.0

    if r_ntc >= THERMISTOR_TABLE[0][0]: return THERMISTOR_TABLE[0][1]
    if r_ntc <= THERMISTOR_TABLE[-1][0]: return THERMISTOR_TABLE[-1][1]

    for i in range(len(THERMISTOR_TABLE) - 1):
        r_high, t_high = THERMISTOR_TABLE[i]
        r_low, t_low   = THERMISTOR_TABLE[i+1]
        if r_ntc <= r_high and r_ntc > r_low:
            fraction = (r_high - r_ntc) / (r_high - r_low)
            return round(t_high + (fraction * (t_low - t_high)), 2)

    return 0.0

def calculate_gsr_resistance_from_raw(raw_val):
    if raw_val >= SERIAL_CALIBRATION: return 0
    try:
        numerator = (32768 + 2 * raw_val) * 10000
        denominator = SERIAL_CALIBRATION - raw_val
        return int(numerator / denominator)
    except:
        return 0

def calculate_accurate_bpm(raw_data, fs):
    if len(raw_data) < 50:
        return 0

    sig = np.array(raw_data)
    sig = sig - np.mean(sig)

    min_dist = int(fs * 0.3)
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=50)
    
    if len(peaks) < 2:
        return 0

    avg_distance = np.mean(np.diff(peaks))
    bpm = 60.0 / (avg_distance / fs)
    return round(bpm, 2)
