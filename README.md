# LogiEdge: Cold-Chain Edge Inference & Automated OTA Deployment System

LogiEdge is an event-driven, edge-AI telemetry and anomaly detection system for cold-chain logistics containers. The pipeline collects multi-modal sensor telemetry (temperature, vibration, and door status) via MQTT, extracts rolling statistical window features, performs INT8 quantized machine learning inference on an edge node (Raspberry Pi 5 / containerized edge daemon), and broadcasts real-time risk alerts. 

The production deployment is orchestrated via **Ansible** playbooks, ensuring automated Over-The-Air (OTA) updates, model synchronization, drift baseline provisioning, and idempotent infrastructure management.

---

## Architecture Overview

```text
 [ Sensor Simulator ]
  (Temp @ 1Hz, Vib @ 0.5Hz, Door)
            │
            ▼ (MQTT Port 1883)
┌─────────────────────────────────┐
│     Mosquitto MQTT Broker       │ ◄─────── Ansible Playbook (logibridge_deploy.yml)
└─────────────────────────────────┘           - Provisioning /opt/logibridge/
            │                                 - Deploying model & reference_dist.json
            ▼ (Topic: .../sensors/#)          - Managing Docker Container Lifecycle
┌────────────────────────────────────────────────────────┐
│     LogiEdge Inference Container (Docker Edge Node)    │
│  - 30s Sliding Window Buffer & Moving Avg Filter       │
│  - Z-Score Normalization (training_stats.npy)          │
│  - Float32 to INT8 Quantization                        │
│  - TFLite Interpreter (M2 INT8 Quantized Model)        │
│  - Population Stability Index (PSI) Drift Monitor      │
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
│   ├── generate_dataset.py      # Feature extraction & dataset generator
│   └── training_stats.npy       # Mean and Std normalization parameters
├── training/
│   ├── train_model.py           # Keras MLP model training script (M1)
│   ├── baseline_model.h5        # Trained FP32 model (Legacy Keras format)
│   ├── X_data.npy               # Training features matrix
│   └── y_data.npy               # Target labels vector
├── optimisation/
│   ├── benchmark.py             # PTQ Quantization & Pruning script (M2 & M3)
│   ├── model_m2_ptq.tflite      # Production INT8 Post-Training Quantized Model
│   └── model_m3_pruned_ptq.tflite # 35% Pruned + INT8 Quantized Model
├── monitoring/
│   └── reference_dist.json      # Baseline data distribution for PSI drift tracking
├── inference/
│   └── inference_service.py     # Edge inference service worker
├── docker/
│   └── Dockerfile               # Production container image manifest
├── deployment/
│   └── logibridge_deploy.yml    # Ansible deployment playbook
├── hosts.ini                    # Ansible inventory file (localhost / remote nodes)
├── requirements.txt             # Python dependencies manifest
└── README.md                    # System documentation
```

---

## Prerequisites & System Requirements

### Hardware Requirements
- **Development Host:** Windows 10/11 with WSL 2 (Ubuntu), macOS, or Linux x86_64 host.
- **Target Edge Node:** Raspberry Pi 5 (8GB recommended) or Docker-compatible Single Board Computer.

### Software Requirements
- **OS Environment:** WSL 2 (Ubuntu 22.04 LTS / 24.04 LTS).
- **Python:** Version 3.11+.
- **Docker Engine / Docker Desktop:** Version 24.0+ with **WSL 2 Integration** enabled.
- **Ansible:** Version 2.14+ with `community.docker` collection.
- **Mosquitto MQTT Broker:** Version 2.0+ running natively in WSL.

---

## Step 1: Environment Setup & Python Virtual Environment

Modern Ubuntu distributions enforce **PEP 668**, preventing system-wide `pip` installs. Always use a dedicated Python virtual environment.

1. **Install Virtual Environment Tools in WSL Ubuntu:**
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-full mosquitto mosquitto-clients
   ```

2. **Navigate to Project Root & Create Virtual Environment:**
   ```bash
   cd /mnt/c/Users/<Your-User>/PycharmProjects/logibridge
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Step 2: Mosquitto MQTT Broker Configuration (WSL)

Mosquitto v2.0+ restricts remote and unauthenticated access by default. Configure the broker to allow local simulation connections:

1. **Configure Listener Settings:**
   ```bash
   echo -e "listener 1883\nallow_anonymous true" | sudo tee /etc/mosquitto/conf.d/local.conf
   ```

2. **Start Mosquitto Service:**
   ```bash
   sudo service mosquitto start
   ```

3. **Verify Service Status:**
   ```bash
   sudo service mosquitto status
   ```

---

## Step 3: Ansible Setup & Passwordless Sudo Configuration

Ansible orchestrates the deployment process locally via WSL or remotely over SSH.

### 1. Install Ansible & Docker Collection
Install Ansible via APT and fetch the Docker collection:
```bash
sudo apt update
sudo apt install -y ansible
ansible-galaxy collection install community.docker
```

### 2. Enable Passwordless Sudo for Local Execution
Because Ansible targets local directories (`/opt/logibridge`) requiring `sudo` privileges, configure passwordless sudo to prevent interactive timeout failures:

```bash
sudo visudo
```
Add the following line at the very bottom of the file (replace `lokesh` with your WSL username):
```text
lokesh ALL=(ALL) NOPASSWD: ALL
```
*Save with `Ctrl + O`, `Enter`, and exit with `Ctrl + X`.*

---

## Step 4: Model Pipeline & Optimization Workflow

Before running the deployment pipeline, build the dataset, baseline model, and quantized artifacts.

1. **Generate Dataset & Feature Statistics:**
   ```bash
   python data_pipeline/generate_dataset.py
   ```
   *Generates `training/X_data.npy`, `training/y_data.npy`, and `data_pipeline/training_stats.npy`.*

2. **Train Baseline FP32 Model (M1):**
   ```bash
   python training/train_model.py
   ```
   *Generates `training/baseline_model.h5` with $\ge 88\%$ validation accuracy target.*

3. **Quantize & Prune Models (M2 & M3):**
   ```bash
   python optimisation/benchmark.py
   ```
   *Generates `optimisation/model_m2_ptq.tflite` (Full INT8) and `optimisation/model_m3_pruned_ptq.tflite` (35% Pruned + INT8).*

4. **Create Drift Reference Distribution:**
   ```bash
   cat << 'EOF' > monitoring/reference_dist.json
   {
     "bins": [0.0, 0.25, 0.5, 0.75, 1.0],
     "expected_probabilities": [0.02, 0.05, 0.13, 0.80]
   }
   EOF
   ```

---

## Step 5: Build Docker Container Image

Build the inference container image from your project root:

```bash
docker build -f docker/Dockerfile -t localhost:5000/logibridge_inference:latest .
```

*Verify the image is in your local Docker cache:*
```bash
docker images | grep logibridge_inference
```

---

## Step 6: Automated Ansible Deployment Pipeline (Task E2)

### 1. Inventory Configuration (`hosts.ini`)
Create or verify `hosts.ini` in the project root:
```ini
[truck_edge_nodes]
localhost ansible_connection=local
```

### 2. Playbook Configuration (`deployment/logibridge_deploy.yml`)
The Ansible playbook provisions runtime paths, deploys models, copies drift benchmarks, and manages container state:

```yaml
---
- name: LogiEdge OTA Deployment and Setup
  hosts: truck_edge_nodes
  become: yes
  vars:
    app_dir: /opt/logibridge
    model_file: "{{ playbook_dir }}/../optimisation/model_m2_ptq.tflite"
    image_name: localhost:5000/logibridge_inference:latest

  tasks:
    - name: Ensure LogiEdge directory exists
      ansible.builtin.file:
        path: "{{ app_dir }}"
        state: directory
        mode: '0755'

    - name: Copy TFLite model to edge node
      ansible.builtin.copy:
        src: "{{ model_file }}"
        dest: "{{ app_dir }}/model.tflite"
        mode: '0644'

    - name: Copy drift monitor reference distribution
      ansible.builtin.copy:
        src: "{{ playbook_dir }}/../monitoring/reference_dist.json"
        dest: "{{ app_dir }}/reference_dist.json"
        mode: '0644'

    - name: Ensure inference container is running with new model
      community.docker.docker_container:
        name: logibridge_inference
        image: "{{ image_name }}"
        state: started
        pull: never
        recreate: true
        restart_policy: always
        etc_hosts:
          host.docker.internal: "host-gateway"
        env:
          MODEL_PATH: "/opt/logibridge/data/model.tflite"
          MQTT_BROKER: "host.docker.internal"
        volumes:
          - "{{ app_dir }}:/opt/logibridge/data"

    - name: Verify container is actively running
      ansible.builtin.command: docker inspect -f '{{ "{{ .State.Running }}" }}' logibridge_inference
      register: verify_running
      changed_when: false
      failed_when: verify_running.stdout != 'true'
```

### 3. Execute Ansible Deployment
Run the playbook from the root project directory:
```bash
ansible-playbook -i hosts.ini deployment/logibridge_deploy.yml
```

### 4. Idempotency Verification (Task E2 Requirement)
Run the exact same command immediately after execution:
```bash
ansible-playbook -i hosts.ini deployment/logibridge_deploy.yml
```
*Verification:* The `PLAY RECAP` must show `changed=0` (`ok=5 changed=0 unreachable=0 failed=0`), confirming complete deployment idempotency.

---

## Step 7: End-to-End System Execution & Monitoring

To run end-to-end telemetry streaming and model inference, open **three independent WSL terminal windows**:

### Terminal 1: Monitor Container Inference Logs
```bash
docker logs -f logibridge_inference
```

### Terminal 2: Run Sensor Telemetry Simulator
Activate your virtual environment and launch the simulator script:
```bash
source venv/bin/activate
python3 data_pipeline/simulator.py --anomaly combined
```

### Terminal 3: Subscribe to MQTT Inference Topic
Inspect real-time risk predictions and drift metrics:
```bash
mosquitto_sub -h localhost -t "logibridge/trucks/+/inference" -v
```

---

## MQTT Topic Schema Reference

| Topic | Publisher | Format | Description |
| :--- | :--- | :--- | :--- |
| `logibridge/trucks/{id}/sensors/temperature` | `simulator.py` | JSON (`{"timestamp": float, "value": float}`) | 1 Hz ambient temperature (°C) |
| `logibridge/trucks/{id}/sensors/vibration` | `simulator.py` | JSON (`{"timestamp": float, "value": float}`) | 0.5 Hz compressor vibration ($g$) |
| `logibridge/trucks/{id}/sensors/door` | `simulator.py` | JSON (`{"timestamp": float, "state": "OPEN"\|"CLOSE"}`) | Discrete door events |
| `logibridge/trucks/{id}/inference` | Container | JSON (`{"timestamp": float, "class": int, "confidence": float}`) | Inference classification output |

---

## Troubleshooting Guide

- **`sudo: interactive authentication is required` / Sudo Timeout in Ansible**
  - *Cause:* Ansible requires `sudo` privileges for `/opt/logibridge` but cannot prompt for a password.
  - *Fix:* Configure passwordless sudo in `/etc/sudoers` via `sudo visudo` (`username ALL=(ALL) NOPASSWD: ALL`).

- **`Could not find or access 'optimisation/model_m2_ptq.tflite'`**
  - *Cause:* Ansible resolves paths relative to the playbook file directory (`deployment/`).
  - *Fix:* Use `{{ playbook_dir }}/../optimisation/model_m2_ptq.tflite` in `logibridge_deploy.yml`.

- **`Unsupported parameters for (community.docker.docker_container) module: source / extra_hosts`**
  - *Cause:* Invalid Ansible module parameters for `docker_container`.
  - *Fix:* Use `pull: never` instead of `source: local`, and `etc_hosts:` instead of `extra_hosts:`.

- **`ConnectionRefusedError: [Errno 111] Connection refused` in Container**
  - *Cause:* Container trying to reach Mosquitto broker on `localhost` inside container loopback.
  - *Fix:* Map `host.docker.internal: "host-gateway"` via `etc_hosts` in the playbook and set `MQTT_BROKER="host.docker.internal"`.

- **`externally-managed-environment` (PEP 668)**
  - *Cause:* Attempting global system `pip install` on modern Ubuntu.
  - *Fix:* Create and activate a Python virtual environment (`python3 -m venv venv && source venv/bin/activate`).