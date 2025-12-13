# HYDRAI

HYDRAI is a smart hydration monitoring system that leverages AI, wearable data (Apple Health), and environmental sensors to provide personalized hydration recommendations. The system uses a FastAPI backend, Deep Learning models (NormWear), and supports both hardware integration (ESP32) and software-only simulation.

## 📂 File Structure

The repository is organized as follows:

* **`backend/`**: The core of the application.
  * **`app/`**: Contains the FastAPI source code (`main.py`, `routes.py`), database logic, and the `dummy_esp_data` simulation logic.
  * **`NormWear/`**: Contains the deep learning models and baseline models for hydration prediction.
  * **`hydr_ai.db`**: The SQLite database storing sensor and user data.
  * **`Dockerfile`**: Configuration for containerizing the backend.
  * **`requirements.txt`**: List of Python dependencies.
* **`CloudScript/`**: Contains `cloud_HRCalc.py`, a bridge script used to process heart rate data and communicate between data sources.
* **`esp32/`**: Contains `main.py`, the MicroPython firmware for the ESP32 microcontroller to handle sensor readings and WiFi communication.
* **`ml_model/`**: Scripts used for training and testing the machine learning models (`train_model.py`, `hydrationdataset.py`).

---

## ⚙️ Setup & Prerequisites

**⚠️ CRITICAL STEP:** Before running any part of the system, you must install the required dependencies and configure your environment.

### 1. Install Dependencies
Navigate to the `backend` folder and install the requirements:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration
You must create an environment setup (e.g., a `.env` file or system variables) containing the following API keys and configurations:

* **`OPENWEATHERAPP_API`**: Key for fetching weather data.
* **`POE_API`**: Key for the AI text generation service.
* **`NTFY_TOPIC`**: The topic name you want to use on [ntfy.sh](https://ntfy.sh) for notifications.

---

## 🚀 How to Run: Scenario A (Hardware/ESP32)
Use this method if you have the physical circuit constructed.

1. **Circuit Setup**: Ensure your ESP32 circuit is built according to the project diagram.
2. **CloudScript & IP**:
   * Run the script located in `CloudScript/cloud_HRCalc.py`.
   * **Important**: Update the frequency for recommendation values within the script as needed.
   * Ensure this script is accessible via your network (not just localhost) to identify your Local IP address.
3. **ESP32 Configuration**:
   * Open `esp32/main.py`.
   * Update the **WiFi Credentials** (SSID/Password).
   * Update the **Target IP** with the Local IP address identified in the previous step.
   * Flash `main.py` to your ESP32. It will begin reading sensors and sending data to the backend.
4. **Apple Health Automation**:
   * Set up the **Apple Health Export** app automation on your iOS device.
   * Configure it to send user data to your CloudScript endpoint at a frequency of **1 minute**.
5. **Initialization**: The CloudScript will forward sensor data to the Google Cloud Container.
   * *Note*: The first transmission may take a moment as the container and internal database initialize.

---

## 💻 How to Run: Scenario B (No Hardware/Dummy Data)
We have configured a simulation mode that allows you to test the entire backend pipeline without physical sensors or an Apple Watch.

**The Logic**: The backend includes a `dummy_esp_data` function. If the server receives an empty JSON payload (`{}`) for sensor or health endpoints, it automatically generates data within a realistic range.

### Option 1: Local Backend Testing (Split Terminal)
This allows you to see the database updates and server logs in real-time on your machine.

**1. Start the Server (Terminal 1)**
Navigate to the backend folder and start the FastAPI server:

```bash
cd backend
uvicorn app.main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

**2. Send Dummy Data (Terminal 2 - PowerShell)**
Open a second terminal window (PowerShell recommended for Windows) and use the following commands to simulate data flow.

* **Insert Dummy Sensor Data:**
  ```powershell
  curl.exe -X POST http://127.0.0.1:8000/sensor -H "Content-Type: application/json" -d "{}"
  ```
* **Insert Dummy Apple-Health Data:**
  ```powershell
  curl.exe -X POST http://127.0.0.1:8000/apple-health -H "Content-Type: application/json" -d "{}"
  ```
* **Get Hydration Recommendation:**
  Once the data sets are populated, request a recommendation:
  ```powershell
  curl.exe http://127.0.0.1:8000/recommendation
  ```

  The terminal will receive JSON responses revealing the status and the calculated values.

### Option 2: Google Cloud Backend Testing
If you want to test the deployed Google Cloud instance:

1. Open your terminal.
2. Run the `curl` commands mentioned above, but replace `http://127.0.0.1:8000` with the actual **Google Cloud Container URL**: `https://hydr-ai-backend-529883695650.us-central1.run.app`.
3. Send empty JSONs (`"{}"`) to the `@apple-health` and `@sensor` endpoints to trigger the dummy data generation.
4. Call the `@recommendation` endpoint to verify the output.

---

## 🔔 Notifications
To receive hydration alerts on your phone:

1. Download the **ntfy.sh** app.
2. Subscribe to the **topic** you defined in your environment variables.
3. You will receive push notifications whenever the backend triggers an alert based on the recommendation logic.
