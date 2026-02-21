# 🏥 Insurance Premium Prediction API

> A production-ready machine learning REST API built with **FastAPI** that predicts insurance premium categories based on user demographics, lifestyle, and socioeconomic features — with full confidence scoring and probability distributions.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Sample Request & Response](#sample-request--response)
- [Model Details](#model-details)
- [Future Roadmap](#future-roadmap)

---

## Overview

This project demonstrates an **end-to-end ML deployment pipeline** — from raw user input to real-time predictions served via a RESTful API. The system ingests demographic and lifestyle data, applies automated feature engineering (BMI computation, lifestyle risk scoring, age bucketing, city-tier classification), and returns a premium category prediction with calibrated confidence scores.

Built with software engineering best practices including **input validation**, **schema enforcement**, **containerization**, and **health-check endpoints** for cloud-native deployment.

---

## Key Features

| Feature | Description |
|---|---|
| **Real-time Inference** | Sub-second predictions via optimized scikit-learn model served with Uvicorn ASGI server |
| **Automated Feature Engineering** | Raw inputs (age, height, weight, city, smoker status) are transformed into model-ready features at inference time |
| **Confidence Scoring** | Returns predicted class alongside full probability distribution across all premium categories |
| **Robust Input Validation** | Pydantic v2 models with field constraints, computed fields, and custom validators |
| **Production-Ready** | Health check endpoint, Docker support, model versioning, and structured error handling |
| **City-Tier Intelligence** | Automatic classification of 50+ Indian cities into Tier-1, Tier-2, and Tier-3 categories |
| **Interactive API Docs** | Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Request                          │
│              (JSON: age, weight, height, city, ...)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Pydantic v2 Input Validation                  │  │
│  │  • Field constraints (age: 0-120, height: 0-2.5m)         │  │
│  │  • City name normalization (.strip().title())              │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Automated Feature Engineering                    │  │
│  │  • BMI = weight / height²                                  │  │
│  │  • lifestyle_risk = f(smoker, BMI)                         │  │
│  │  • age_group = f(age) → young/adult/middle_aged/senior     │  │
│  │  • city_tier = f(city) → 1 / 2 / 3                        │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │             scikit-learn Model Inference                    │  │
│  │  • Predict class label                                     │  │
│  │  • Predict probabilities for all classes                   │  │
│  │  • Return confidence score (max probability)               │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │                                          │
└───────────────────────┼──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                     JSON Response                                │
│  { predicted_category, confidence, class_probabilities }         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.115 | High-performance async REST API with auto-generated OpenAPI docs |
| **ML Framework** | scikit-learn 1.6.1 | Trained classification model with `predict` & `predict_proba` |
| **Data Validation** | Pydantic v2.11 | Schema enforcement, computed fields, custom validators |
| **Data Processing** | Pandas 2.2.3, NumPy 2.2.6 | DataFrame construction for model-compatible input |
| **Server** | Uvicorn 0.34 | Lightning-fast ASGI server |
| **Containerization** | Docker (Python 3.11-slim) | Reproducible, lightweight deployment |
| **Serialization** | Pickle | Model persistence and loading |

---

## Project Structure

```
insurance-premium-prediction-fastapi/
│
├── app.py                          # FastAPI application entry point & route definitions
├── Dockerfile                      # Container configuration for deployment
├── requirements.txt                # Pinned Python dependencies
│
├── model/
│   ├── model.pkl                   # Serialized scikit-learn classification model
│   └── predict.py                  # Model loading, inference logic & confidence scoring
│
├── schema/
│   ├── user_input.py               # Pydantic input schema with feature engineering
│   └── prediction_response.py      # Pydantic response schema with field documentation
│
└── config/
    └── city_tier.py                # City classification data (Tier-1, Tier-2, Tier-3)
```

---

## Feature Engineering

The API performs **real-time feature engineering** at inference time using Pydantic computed fields — no separate preprocessing pipeline needed:

### 1. BMI Calculation
```python
bmi = weight / (height ** 2)
```

### 2. Lifestyle Risk Scoring
| Condition | Risk Level |
|---|---|
| Smoker **AND** BMI > 30 | `high` |
| Smoker **OR** BMI > 27 | `medium` |
| Otherwise | `low` |

### 3. Age Group Bucketing
| Age Range | Category |
|---|---|
| < 25 | `young` |
| 25 – 44 | `adult` |
| 45 – 59 | `middle_aged` |
| ≥ 60 | `senior` |

### 4. City Tier Classification
| Tier | Cities |
|---|---|
| **Tier 1** | Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune |
| **Tier 2** | Jaipur, Chandigarh, Indore, Lucknow, + 40 more cities |
| **Tier 3** | All other cities (default fallback) |

> **Design Decision:** Feature engineering is embedded inside the Pydantic schema using `@computed_field`, ensuring that raw user-facing inputs are automatically transformed into model-ready features — eliminating training-serving skew.

---

## API Endpoints

### `GET /` — Home
Returns a welcome message confirming the API is running.

### `GET /health` — Health Check
Production-grade health endpoint for load balancers and orchestrators (e.g., AWS ALB, Kubernetes).

```json
{
  "status": "OK",
  "version": "1.0.0",
  "model_loaded": true
}
```

### `POST /predict` — Predict Premium Category
Accepts user demographic data and returns insurance premium prediction with confidence scores.

---

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/insurance-premium-prediction-fastapi.git
cd insurance-premium-prediction-fastapi

# Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the API server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Docker Deployment

```bash
# Build the image
docker build -t insurance-premium-api .

# Run the container
docker run -d -p 8000:8000 --name premium-predictor insurance-premium-api
```

The Docker image uses `python:3.11-slim` for a minimal ~150MB footprint, suitable for deployment on **AWS ECS**, **Google Cloud Run**, **Azure Container Apps**, or **Kubernetes**.

---

## Sample Request & Response

### Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "weight": 85.5,
    "height": 1.75,
    "income_lpa": 12.5,
    "smoker": false,
    "city": "Mumbai",
    "occupation": "private_job"
  }'
```

### Response
```json
{
  "response": {
    "predicted_category": "Medium",
    "confidence": 0.7523,
    "class_probabilities": {
      "Low": 0.1204,
      "Medium": 0.7523,
      "High": 0.1273
    }
  }
}
```

### Key Response Fields

| Field | Type | Description |
|---|---|---|
| `predicted_category` | `string` | Most likely premium tier (e.g., Low, Medium, High) |
| `confidence` | `float` | Model's confidence in the prediction (0.0 – 1.0) |
| `class_probabilities` | `dict` | Full probability distribution across all premium categories |

---

## Model Details

| Attribute | Value |
|---|---|
| **Framework** | scikit-learn 1.6.1 |
| **Model Type** | Classification (with `predict_proba` support) |
| **Model Version** | 1.0.0 |
| **Serialization** | Python Pickle |
| **Input Features** | `bmi`, `age_group`, `lifestyle_risk`, `city_tier`, `income_lpa`, `occupation` |
| **Output** | Multi-class premium category with probability calibration |

### Input Feature Schema

| Feature | Type | Source |
|---|---|---|
| `bmi` | `float` | Computed: `weight / height²` |
| `age_group` | `str` | Computed: age → categorical bucket |
| `lifestyle_risk` | `str` | Computed: `f(smoker, bmi)` |
| `city_tier` | `int` | Computed: city name → tier lookup |
| `income_lpa` | `float` | Direct user input |
| `occupation` | `str` | Direct user input (7 categories) |

---

<!-- ## Future Roadmap

- [ ] **MLflow Integration** — Experiment tracking, model registry, and automated versioning
- [ ] **CI/CD Pipeline** — GitHub Actions for automated testing and deployment
- [ ] **A/B Testing** — Shadow mode deployment for model comparison
- [ ] **Model Monitoring** — Data drift detection and prediction logging
- [ ] **Rate Limiting** — API throttling for production traffic management
- [ ] **Authentication** — JWT/OAuth2 secured endpoints
- [ ] **Batch Prediction** — Bulk inference endpoint for CSV/JSON uploads
- [ ] **Model Explainability** — SHAP/LIME integration for prediction interpretability
- [ ] **Cloud Deployment** — AWS ECS / GCP Cloud Run Terraform configs -->

<!-- --- -->

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ for the intersection of Machine Learning and Software Engineering</b>
</p>
