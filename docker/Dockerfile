FROM python:3.11-slim

WORKDIR /opt/logibridge

# 1. Install dependencies first to cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy inference script
COPY inference/inference_service.py .

# 3. Copy normalization stats required by inference script
COPY data_pipeline/training_stats.npy .

# 4. Copy model last to maximize layer caching during OTA updates
# Renaming it to model.tflite in the destination container
COPY optimisation/model_m2_ptq.tflite ./model.tflite

ENV MODEL_PATH="/opt/logibridge/model.tflite"

CMD ["python", "inference_service.py"]
