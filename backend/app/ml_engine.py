import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

def predict_hydration_level(hr, temp, gsr, humidity, outside_temp, steps):
    x = np.array([[hr, temp, gsr, humidity, outside_temp, steps]])
    label = model.predict(x)[0]
    categories = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    return categories[label]
