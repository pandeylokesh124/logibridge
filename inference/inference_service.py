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

MODEL_PATH = os.getenv("MODEL_PATH", "model.tflite")
STATS_PATH = os.getenv("STATS_PATH", "training_stats.npy")

# Fallback check for local host execution outside Docker
if not os.path.exists(MODEL_PATH) and os.path.exists("../optimisation/model_m2_ptq.tflite"):
    MODEL_PATH = "../optimisation/model_m2_ptq.tflite"

if not os.path.exists(STATS_PATH) and os.path.exists("../data_pipeline/training_stats.npy"):
    STATS_PATH = "../data_pipeline/training_stats.npy"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'")

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Stats file not found at '{STATS_PATH}'")

# Buffers for the 30-second sliding window
temp_buffer = deque(maxlen=30)
vib_buffer = deque(maxlen=15)
last_inference_time = time.time()

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


def on_message(client, userdata, msg):
    global last_inference_time
    payload = json.loads(msg.payload.decode())

    if "temperature" in msg.topic:
        temp_buffer.append(payload["value"])
    elif "vibration" in msg.topic:
        vib_buffer.append(payload["value"])

    current_time = time.time()
    if len(temp_buffer) == 30 and len(vib_buffer) == 15 and (current_time - last_inference_time >= 10):
        last_inference_time = current_time

        # Preprocess features
        features = extract_features(list(temp_buffer), list(vib_buffer))
        input_data = np.expand_dims(features, axis=0)

        # Quantize Float32 input to INT8
        input_scale, input_zero_point = input_details[0]['quantization']
        if input_scale > 0:
            input_data = np.clip(np.round(input_data / input_scale + input_zero_point), -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(np.int8)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dequantize INT8 output back to Float32
        output_scale, output_zero_point = output_details[0]['quantization']
        if output_scale > 0:
            output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
        else:
            output_float = output_data.astype(np.float32)

        predicted_class = int(np.argmax(output_float))
        confidence = float(np.max(output_float))

        result = {
            "timestamp": current_time,
            "class": predicted_class,
            "confidence": confidence
        }
        client.publish(f"logibridge/trucks/{TRUCK_ID}/inference", json.dumps(result), qos=1)
        print(f"Inference published: Class {predicted_class} ({confidence:.2f})")


def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)

    client.subscribe(f"logibridge/trucks/{TRUCK_ID}/sensors/temperature")
    client.subscribe(f"logibridge/trucks/{TRUCK_ID}/sensors/vibration")

    print(f"LogiEdge Inference Service started using model: {MODEL_PATH}")
    print("Listening for sensor data...")
    client.loop_forever()


if __name__ == "__main__":
    main()
