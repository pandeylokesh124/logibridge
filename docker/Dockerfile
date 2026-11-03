FROM python:3.11-slim

WORKDIR /opt/logibridge
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files from their exact host directories into the container workdir
COPY inference/inference_service.py .
COPY data_pipeline/training_stats.npy .
COPY optimisation/model_m2_ptq.tflite ./model.tflite

ENV MODEL_PATH="/opt/logibridge/model.tflite"
ENV STATS_PATH="/opt/logibridge/training_stats.npy"
ENV MQTT_BROKER="host.docker.internal"

CMD ["python", "inference_service.py"]