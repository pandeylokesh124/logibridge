import os
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from collections import deque

# Configuration
TRUCK_ID = "TRK-PH-017"
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")

# Force absolute container paths as primary locations
MODEL_PATH = "/opt/logibridge/model.tflite"
STATS_PATH = "/opt/logibridge/training_stats.npy"

# Fall back to environment variables if absolute paths aren't found (e.g., local testing)
if not os.path.exists(MODEL_PATH) and os.getenv("MODEL_PATH"):
    MODEL_PATH = os.getenv("MODEL_PATH")

if not os.path.exists(STATS_PATH) and os.getenv("STATS_PATH"):
    STATS_PATH = os.getenv("STATS_PATH")

# Final verification checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'")

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Stats file not found at '{STATS_PATH}'")

# Buffers for the sliding window
temp_buffer = deque(maxlen=30)
vib_buffer = deque(maxlen=15)
last_inference_time = time.time()

temp_count = 0
vib_count = 0

# Load Model and Preprocessing Stats
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
stats = np.load(STATS_PATH, allow_pickle=True).item()


def extract_features(temp_window, vib_window):
    t_smooth = np.convolve(temp_window, np.ones(5) / 5, mode='valid')
    v_smooth = np.convolve(vib_window, np.ones(5) / 5, mode='valid')

    t_mean = np.mean(t_smooth)
    t_std = np.std(t_smooth)
    t_roc = (t_smooth[-1] - t_smooth[0]) / (len(t_smooth) / 60.0)

    v_rms = np.sqrt(np.mean(v_smooth ** 2))
    v_peak = np.max(np.abs(v_smooth))
    v_kurt = np.mean((v_smooth - np.mean(v_smooth)) ** 4) / (np.std(v_smooth) ** 4 + 1e-6)

    features = np.array([t_mean, t_std, t_roc, v_rms, v_peak, v_kurt], dtype=np.float32)
    return (features - stats['mean']) / stats['std']


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to MQTT Broker successfully with code: {rc}")
    # Subscribe to all logibridge topics to guarantee message capture
    client.subscribe("logibridge/#")
    print("Subscribed to topic wildcard: logibridge/#")


def on_message(client, userdata, msg):
    global last_inference_time, temp_count, vib_count

    print(f"[RECV] Topic: {msg.topic}")

    try:
        payload = json.loads(msg.payload.decode())
    except Exception as e:
        print(f"Failed to parse payload on {msg.topic}: {e}")
        return

    topic_lower = msg.topic.lower()
    val = payload.get("value") if isinstance(payload, dict) else payload

    if val is not None:
        if "temperature" in topic_lower or "temp" in topic_lower:
            temp_buffer.append(float(val))
            temp_count += 1
        elif "vibration" in topic_lower or "vib" in topic_lower:
            vib_buffer.append(float(val))
            vib_count += 1

    current_time = time.time()

    # Run inference when buffers are filled and 10 seconds have elapsed
    if len(temp_buffer) == 30 and len(vib_buffer) == 15:
        if current_time - last_inference_time >= 10:
            last_inference_time = current_time

            # Feature Extraction
            features = extract_features(list(temp_buffer), list(vib_buffer))
            input_data = np.expand_dims(features, axis=0)

            # Quantize Float32 input to INT8
            input_scale, input_zero_point = input_details[0]['quantization']
            if input_scale > 0:
                input_data = np.clip(np.round(input_data / input_scale + input_zero_point), -128, 127).astype(np.int8)
            else:
                input_data = input_data.astype(np.int8)

            # Model Execution with Latency Tracking
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Dequantize INT8 output back to Float32
            output_scale, output_zero_point = output_details[0]['quantization']
            if output_scale > 0:
                output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                output_float = output_data.astype(np.float32)

            predicted_class = int(np.argmax(output_float))
            confidence = float(np.max(output_float))

            # Alert Triggers
            if predicted_class == 2:
                print(f"[CRITICAL ALERT] Severe anomaly on {TRUCK_ID}! Confidence: {confidence:.2f} | Latency: {latency_ms:.2f}ms. Notifying center...")
                alert_payload = {
                    "truck_id": TRUCK_ID,
                    "severity": "CRITICAL",
                    "timestamp": current_time,
                    "confidence": confidence,
                    "action_required": "Inspect cooling unit & driver contact immediately."
                }
                client.publish(f"logibridge/trucks/{TRUCK_ID}/alerts/critical", json.dumps(alert_payload), qos=2)

            elif predicted_class == 1:
                print(f"[WARNING] Anomaly detected on {TRUCK_ID}. Confidence: {confidence:.2f} | Latency: {latency_ms:.2f}ms. Notify driver ...")
                warning_payload = {
                    "truck_id": TRUCK_ID,
                    "severity": "WARNING",
                    "timestamp": current_time,
                    "confidence": confidence,
                    "action_required": "Schedule routine maintenance check."
                }
                client.publish(f"logibridge/trucks/{TRUCK_ID}/alerts/warning", json.dumps(warning_payload), qos=1)

            # Publish Standard Prediction Payload
            result = {
                "timestamp": current_time,
                "class": predicted_class,
                "confidence": confidence,
                "latency_ms": round(latency_ms, 2)
            }
            client.publish(f"logibridge/trucks/{TRUCK_ID}/inference", json.dumps(result), qos=1)
            print(f"Inference published: Class {predicted_class} ({confidence:.2f}) in {latency_ms:.2f}ms")


def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, 1883, 60)

    print(f"LogiEdge Inference Service started using model: {MODEL_PATH}")
    print(f"Connected to broker: {MQTT_BROKER}:1883")
    print("Listening for sensor data...")
    client.loop_forever()


if __name__ == "__main__":
    main()
