# License Plate Detection Pipeline with YOLOv8, Airflow, and MLflow

This project is an end-to-end **MLOps pipeline** for **License Plate Detection** using the **YOLOv8** object detection model. The pipeline is orchestrated with **Apache Airflow** and integrates **MLflow** for comprehensive experiment tracking, model versioning, and performance logging.

This project is a part of MLOps cource (CS317.P22), with members:
- Huynh La viet Toan: 22521486
- Nguyen Truong Minh Khoa: 22520680
- Nguyen Thanh Luan: 22520826
- Luong Truong Thinh: 22521412
- Phan Phuoc Loc Ngoc: 22520960


## Key Features

### Automated MLOps Workflow:
- Built using **Apache Airflow** with `@task`-decorated Python functions.

### Experiment Tracking with MLflow
- Tracks **hyperparameters**, **model metrics**, and **checkpoints**.
- Supports nested runs for hyperparameter tuning and evaluation.
- Automatically logs dataset metadata: **source**, **url**, **version**.

### Hyperparameter Tuning & Model Evaluation
- Performs grid search over key training parameters (`freeze_layers`, `epochs`, `lr0`).
- Evaluates multiple model variants and selects the best based on evaluation metrics.
- Stores and logs the **best model** based on customizable metrics.

### Task Orchestration
- Scheduled to run every 30 minutes.
- Task-level retries and modular flow for robust execution.


## How It Works

1. **Initialize Hyperparameters**  
   Generates a grid of parameters to try during training.

2. **Prepare Pretrained Model**  
   Uses an existing YOLOv8 model or trains a base model from scratch and logs it to MLflow.

3. **Train and Validate Models**  
   Trains models using different hyperparameter combinations, logs metrics and weights.

4. **Evaluate Models**  
   Evaluates trained models and selects the best based on a specific metric.

5. **Save Best Model**  
   Logs the final best-performing model and its metadata to MLflow.


## Tech Stack

### Airflow
Apache Airflow is a platform to programmatically author, schedule, and monitor workflows. It's widely used for orchestrating complex data pipelines with built-in retry logic and rich UI.

**Highlights:**
- DAG-based workflow orchestration
- Task dependencies and scheduling
- Scalable and extensible with plugins
- Integration with many services (e.g., AWS, GCP)

### MLflow
MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.

**Highlights:**
- Experiment tracking with metrics and parameters
- Model packaging and versioning
- Integration with many ML libraries and tools
- Supports model deployment via REST API

### Ultralytics
Ultralytics develops cutting-edge models for real-time object detection, including YOLOv5 and YOLOv8, optimized for performance and usability.

**Highlights:**
- High-performance object detection models
- Simple training and inference APIs
- Pretrained models and custom training support
- Export to multiple formats (ONNX, CoreML, etc.)

### Docker
Docker is a platform for developing, shipping, and running applications in lightweight, portable containers.

**Highlights:**
- Environment consistency across development and production
- Containerization of ML models and pipelines
- Easy integration with CI/CD workflows
- Scalable deployment with Docker Compose or Kubernetes


## Requirements

- Python 3.10+
- Apache Airflow 2.10.5
- MLflow 2.21.2
- Dependencies: ultralytics 8.3.107 (for YOLOv8)
- Docker


## Setup & Usage

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/luanntd/License-Plate-Detection-Pipeline-with-Experiment-Tracking.git
    cd License-Plate-Detection-Pipeline-with-Experiment-Tracking
    ```
2.  **Prepare Dataset:**
    - You can download dataset used for this project from [this url](https://www.kaggle.com/datasets/bomaich/vnlicenseplate) and paste the **train**, **valid**, **test** folders in `src/dataset/`.
    - Or you can choose a different dataset and follow the `README.md` in `src/dataset/`.

<!-- 2.  **Set Up Environment Variables:**
    * Create a file named `.env` in the project's root directory.
    * Run the following command to create airflow user:

        ```bash
        echo -e "AIRFLOW_UID=$(id -u)" > .env
        ``` -->

3.  **Running Docker containers:**
    ```bash
    docker-compose up -d
    ```
  
4.  **Access the UI:**
    - Open your web browser and navigate to http://localhost:8080 for Airflow webserver. Log in via **username**: `airflow`, **password**: `airflow`.
    - Open your web browser and navigate to http://localhost:5000 for MLflow UI.


## Demo

### Airflow Pipeline

![airflow_pipeline](assets/airflow_pipeline.gif)
![completed_pipeline](assets/completed_pipeline.png)

### MLflow Experiment Tracking

![mlflow_experiment_tracking](assets/experiment_tracking.gif)

## Collaborators
<a href="https://github.com/luanntd">
  <img src="https://github.com/luanntd.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/Khoa-Nguyen-Truong">
  <img src="https://github.com/Khoa-Nguyen-Truong.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/HuynhToan2004">
  <img src="https://github.com/HuynhToan2004.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/locngocphan12">
  <img src="https://github.com/locngocphan12.png?size=50" width="50" style="border-radius: 50%;" />
</a>
<a href="https://github.com/thinhlt04">
  <img src="https://github.com/thinhlt04.png?size=50" width="50" style="border-radius: 50%;" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>