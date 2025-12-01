import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

np.random.seed(42)
n_samples = 1000

# Features
data = pd.DataFrame({
    "hr": np.random.randint(60, 120, n_samples),
    "temp": np.random.uniform(36, 39, n_samples),
    "gsr": np.random.uniform(0.2, 1.0, n_samples),
    "humidity": np.random.randint(30, 90, n_samples),
    "outside_temp": np.random.uniform(-5, 35, n_samples),
    "steps": np.random.randint(0, 15000, n_samples),
})

# Labels (0=Normal, 1=Mild, 2=Moderate, 3=Severe)
def label_hydration(row):
    score = 0
    if row["hr"] > 100: score += 1
    if row["temp"] > 37.5: score += 1
    if row["gsr"] > 0.7: score += 1
    if row["outside_temp"] > 30 or row["humidity"] < 40: score += 1
    if row["steps"] > 10000: score += 1
    
    if score <= 1: return 0
    elif score == 2: return 1
    elif score == 3: return 2
    else: return 3

data["hydration_level"] = data.apply(label_hydration, axis=1)

# Train/test split
X = data[["hr", "temp", "gsr", "humidity", "outside_temp", "steps"]]
y = data["hydration_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Classifier saved as model.pkl")
