# LogiEdge: Cold-Chain Edge Inference System

LogiEdge is an event-driven, edge-AI telemetry and anomaly detection system for cold-chain logistics containers. The pipeline collects multi-modal sensor telemetry (temperature, vibration, and door status) via MQTT, extracts rolling statistical window features, performs INT8 quantized machine learning inference on a single-board computer / containerized edge node, and broadcasts real-time risk alerts.

---

## Architecture Overview

```
 [ Sensor Simulator ]
  (Temp @ 1Hz, Vib @ 0.5Hz, Door)
            │
            ▼ (MQTT Port 1883)
┌─────────────────────────────────┐
│     Mosquitto MQTT Broker       │
└─────────────────────────────────┘
            │
            ▼ (Topic: .../sensors/#)
┌────────────────────────────────────────────────────────┐
│     LogiEdge Inference Container (Docker Edge Node)    │
│  - 30s Sliding Window Buffer & Moving Avg Filter       │
│  - Z-Score Normalization (training_stats.npy)          │
│  - Float32 to INT8 Quantization                        │
│  - TFLite Interpreter (M2 INT8 Quantized Model)        │
│  - INT8 to Float32 Dequantization & Argmax Softmax     │
└────────────────────────────────────────────────────────┘
            │
            ▼ (Topic: .../inference)
 ┌──────────────────────────────┐
 │   Alerts / Cloud / Dashboard │
 └──────────────────────────────┘
```

---

## Directory Structure

```text
logibridge/
├── data_pipeline/
│   ├── simulator.py             # MQTT sensor data stream generator
│   ├── generate_dataset.py      # Feature extraction & training dataset generator
│   └── training_stats.npy       # Mean and Std normalization parameters
├── training/
│   ├── train_model.py           # Keras MLP model training script (M1)
│   ├── baseline_model.h5        # Trained FP32 model (Legacy Keras format)
│   ├── X_data.npy               # Training features matrix
│   └── y_data.npy               # Target labels vector
├── optimisation/
│   ├── benchmark.py             # Quantization & Pruning script (M2 & M3)
│   ├── model_m2_ptq.tflite      # Full INT8 Post-Training Quantized Model
│   └── model_m3_pruned_ptq.tflite # 35% Pruned + INT8 Quantized Model
├── inference/
│   └── inference_service.py     # Edge inference service worker
├── docker/
│   └── Dockerfile               # Production container image manifest
├── requirements.txt             # Python dependencies manifest
└── README.md                    # System documentation
```

---

## Prerequisites & System Requirements

### Hardware Requirements
* **Development Machine:** Windows 10/11, macOS, or Linux x86_64 host.
* **Target Edge Node:** Raspberry Pi 5 (8GB recommended) or Docker-compatible Single Board Computer.

### Software Requirements
* **Python:** Version 3.11 or 3.12.
* **Docker Engine / Docker Desktop:** Version 24.0+ with `host.docker.internal` DNS routing enabled.
* **Mosquitto MQTT Broker:** Version 2.0+.
* **MQTT Client CLI Tools:** `mosquitto_sub` and `mosquitto_pub` installed locally.

---

## Environment Setup & Dependencies

1. **Clone Repository & Create Environment:**
   ```bash
   git clone https://github.com/pandeylokesh124/logibridge.git
   cd logibridge
   python -m venv venv
   ```

2. **Activate Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   Verify `requirements.txt` contains:
   ```text
   numpy>=1.24.0
   tensorflow>=2.16.1
   tf_keras>=2.16.0
   tensorflow-model-optimization>=0.8.0
   scikit-learn>=1.3.0
   paho-mqtt>=1.6.1
   ```
   Install via pip:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Step 1: Mosquitto MQTT Broker Setup

The system requires an active MQTT broker listening on port `1883` (TCP) and optionally `9001` (WebSockets).

### Method A: Local Windows Service
1. Download and install [Eclipse Mosquitto](https://mosquitto.org/download/).
2. Edit `C:\Program Files\mosquitto\mosquitto.conf`:
   ```ini
   listener 1883
   allow_anonymous true
   ```
3. Start the service:
   ```cmd
   net start mosquitto
   ```

### Method B: Standalone Docker Container
```bash
docker run -d --name mosquitto -p 1883:1883 -p 9001:9001 eclipse-mosquitto:latest
```

### Verification
Test that the broker is listening on port 1883:
* **Windows PowerShell:**
  ```powershell
  Test-NetConnection -ComputerName localhost -Port 1883
  ```
* **Linux / macOS:**
  ```bash
  nc -zv localhost 1883
  ```

---

## Step 2: Dataset Generation & Feature Pipeline

Generate synthetic multi-stream sensor data (Temperature drift, Vibration spikes) and compute 6 sliding-window features:
1. Temperature Moving Average ($T_{\text{mean}}$)
2. Temperature Standard Deviation ($T_{\text{std}}$)
3. Temperature Rate of Change ($T_{\text{roc}}$)
4. Vibration Root Mean Square ($V_{\text{rms}}$)
5. Vibration Peak ($V_{\text{peak}}$)
6. Vibration Kurtosis ($V_{\text{kurt}}$)

Execute dataset generation from project root:
```bash
python training/generate_dataset.py
```
**Outputs Generated:**
* `training/X_data.npy`: Normalized window features array.
* `training/y_data.npy`: Multi-class targets (0: Normal, 1: Warning, 2: Critical).
* `data_pipeline/training_stats.npy`: Feature mean and standard deviation dictionary for Z-score normalization.

---

## Step 3: Baseline Model Training (M1)

Train the FP32 2-hidden-layer MLP (32 and 16 ReLU units, 3-class Softmax output). To avoid Keras 3 deserialization issues with model optimization toolchains, set legacy mode:

```bash
python training/train_model.py
```

**Training Validation Target:** $\ge 88\%$ Accuracy.  
**Output Generated:** `training/baseline_model.h5`

---

## Step 4: Model Optimization & Quantization (M2 & M3)

Convert the baseline Keras model into deployable edge artifacts using TensorFlow Lite Post-Training Quantization (PTQ) and Structured Weight Pruning.

```bash
python optimisation/benchmark.py
```

### Generated Artifact Variants
| Variant | Name | Optimization Strategy | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **M1** | `baseline_model.h5` | FP32 Baseline Keras Model | Development benchmark |
| **M2** | `model_m2_ptq.tflite` | Full INT8 Post-Training Quantization | **Primary Production Deployment** (High recall, 3 KB footprint) |
| **M3** | `model_m3_pruned_ptq.tflite` | 35% Polynomial Pruning + INT8 PTQ | Resource-constrained edge experiments |

---

## Step 5: Docker Containerization

The production edge inference engine is packaged into a minimal Docker container.

### Dockerfile Inspection (`docker/Dockerfile`)
```dockerfile
FROM python:3.11-slim

WORKDIR /opt/logibridge

# Cache dependencies layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application scripts and stats
COPY inference/inference_service.py .
COPY data_pipeline/training_stats.npy .

# Copy production M2 model
COPY optimisation/model_m2_ptq.tflite ./model.tflite

ENV MODEL_PATH="/opt/logibridge/model.tflite"
ENV STATS_PATH="/opt/logibridge/training_stats.npy"
ENV MQTT_BROKER="localhost"

CMD ["python", "inference_service.py"]
```

### Build Docker Image
Execute the build command from the **project root directory**:
```bash
docker build -f docker/Dockerfile -t localhost:5000/logiedge_inference:latest .
```

---

## Step 6: End-to-End System Execution & Verification

To verify full system operations, open **three independent terminal windows**:

### Terminal 1: Run Docker Inference Container
Pass `host.docker.internal` so the container connects to the host machine's Mosquitto broker:
```bash
docker run --rm -it \
  -e MQTT_BROKER="host.docker.internal" \
  -e MODEL_PATH="/opt/logibridge/model.tflite" \
  --name logiedge_inference \
  localhost:5000/logiedge_inference:latest
```
*Expected Terminal Output:*
```text
LogiEdge Inference Service started using model: /opt/logibridge/model.tflite
Listening for sensor data...
```

### Terminal 2: Start Sensor Telemetry Simulator
Run the simulator to broadcast temperature, vibration, and door telemetry:
```bash
python data_pipeline/simulator.py --anomaly none
```

### Terminal 3: Subscribe to MQTT Inference Stream
Listen for real-time risk predictions broadcast by the container:
```bash
mosquitto_sub -h localhost -t "logibridge/trucks/TRK-PH-017/inference" -v
```

### Testing Anomaly Scenarios
In Terminal 2, restart `simulator.py` with different anomaly triggers to verify classification switching:

1. **Normal Telemetry (`--anomaly none`):**
   ```bash
   python data_pipeline/simulator.py --anomaly none
   ```
   *Output in Terminal 3 (after 30s buffer fill):*
   `logibridge/trucks/TRK-PH-017/inference {"timestamp": 1730537400.12, "class": 0, "confidence": 0.98}`

2. **Temperature Drift Anomaly (`--anomaly temp_drift`):**
   ```bash
   python data_pipeline/simulator.py --anomaly temp_drift
   ```
   *Output in Terminal 3:*
   `logibridge/trucks/TRK-PH-017/inference {"timestamp": 1730537410.15, "class": 1, "confidence": 0.94}`

3. **Combined Thermal & Mechanical Failure (`--anomaly combined`):**
   ```bash
   python data_pipeline/simulator.py --anomaly combined
   ```
   *Output in Terminal 3:*
   `logibridge/trucks/TRK-PH-017/inference {"timestamp": 1730537420.18, "class": 2, "confidence": 0.99}`

---

## MQTT Topic Schema Reference

| Topic | Publisher | Format | Description |
| :--- | :--- | :--- | :--- |
| `logibridge/trucks/{id}/sensors/temperature` | `simulator.py` | JSON (`{"timestamp": float, "value": float}`) | 1 Hz ambient temperature (°C) |
| `logibridge/trucks/{id}/sensors/vibration` | `simulator.py` | JSON (`{"timestamp": float, "value": float}`) | 0.5 Hz compressor vibration (g) |
| `logibridge/trucks/{id}/sensors/door` | `simulator.py` | JSON (`{"timestamp": float, "state": "OPEN"\|"CLOSE"}`) | Discrete door status events |
| `logibridge/trucks/{id}/inference` | Container | JSON (`{"timestamp": float, "class": int, "confidence": float}`) | Real-time classification output |

---

## Troubleshooting & Edge Cases

* **ValueError: Cannot set tensor: Got value of type FLOAT64 but expected type INT8**  
  *Cause:* Preprocessed feature array was fed directly to INT8 TFLite model.  
  *Fix:* `inference_service.py` handles input quantization via `(features / input_scale) + input_zero_point` before `interpreter.invoke()`.

* **TypeError / Deserialization Failure on Model Load**  
  *Cause:* Keras 3 vs Keras 2 version mismatch.  
  *Fix:* Ensure `os.environ["TF_USE_LEGACY_KERAS"] = "1"` is set prior to importing `tensorflow` in all training and benchmark scripts, and save model files as `.h5`.

* **Container Cannot Connect to MQTT Broker**  
  *Cause:* Container attempting to reach `localhost` inside its isolated network stack.  
  *Fix:* Pass `-e MQTT_BROKER="host.docker.internal"` on Windows/macOS, or `--net=host` on native Linux environments.