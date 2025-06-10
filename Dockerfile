FROM apache/airflow:slim-2.9.2-python3.10

COPY src src/

USER root

# Install system dependencies for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir mlflow
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir psycopg2-binary
