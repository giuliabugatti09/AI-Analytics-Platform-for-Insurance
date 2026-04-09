# 📸 Smart Insurance: AI-Driven Road Safety & Incident Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.ai/)

## 📝 Executive Summary
This platform is a comprehensive Deep Learning and Machine Learning solution designed to modernize traffic incident management for the insurance sector. Developed as part of a forensic intelligence simulation, the system integrates computer vision, tabular prediction, and geospatial intelligence to transform raw traffic data into actionable safety insights.

### 🏆 Award-Winning Project & Partnership
This project was developed for academic purposes as part of a strategic partnership with **Sompo Seguros**. 
* **Challenge:** FIAP Innovation Challenge (Sompo Seguros).
* **Recognition:** **3rd Place Winner** (2025).
* **Objective:** To demonstrate how Artificial Intelligence can optimize claim triaging and enhance road safety for policyholders.

### 🛡️ Data Privacy & Compliance
All data utilized in this project is **Public Domain**.
* **Source:** Official datasets released by the **Brazilian Government (PRF - Polícia Rodoviária Federal)** via the Access to Information Law.
* **Compliance:** There are no PII (Personally Identifiable Information) or sensitive private insurance data issues. All information is anonymized and legally cleared for academic and research purposes.


## 🚀 Key Modules

### 1. 🔍 Computer Vision Diagnostic
A high-performance **CNN (EfficientNetV2-B0)** trained via **Transfer Learning** to automatically detect road accidents from visual evidence (CCTV or photo uploads). 
- **Tech:** TensorFlow, Keras, Data Augmentation.
- **Goal:** Rapid claim verification and automated emergency alerts.

### 2. 🤖 Predictive Intelligence Hub
A gradient boosting engine (**XGBoost**) that forecasts the severity of incidents based on historical patterns, infrastructure characteristics, and environmental conditions.
- **Tech:** Scikit-Learn, Joblib, Feature Engineering.
- **Input:** PRF (Federal Highway Police) structured data.

### 3. 🌍 Geospatial Risk Mapping
Interactive heatmaps identifying high-density accident hotspots across Brazil, allowing for regional risk assessment and infrastructure auditing.
- **Tech:** PyDeck, cKDTree (Spatial Indexing).

### 4. 💬 Accident Specialist AI
An integrated LLM assistant powered by **Google Gemini** that acts as a forensic consultant, helping analysts interpret accident dynamics and legal implications.

---

## 📂 Project Structure
```text
├── .streamlit/             # Streamlit configuration and secret management
├── .venv/                  # Python virtual environment (isolated dependencies)
├── assets/                 # Raw image datasets, branding icons, and screenshots
├── components/             # Modular UI components (Specialized AI Agent)
├── data/                   # Processed datasets (CSV) and highway metadata
├── models/                 # Serialized AI weights (.keras for CNN, .pkl for XGBoost)
├── notebooks/              # Jupyter notebooks for model training and EDA research
├── pages/                  # Multi-page application modules (EDA, Predictions, Maps)
├── utils/                  # Utility scripts for security (SSRF) and geospatial logic
├── .gitignore              # Instructions for Git to ignore temporary/sensitive files
├── Home.py                 # Application landing page and entry point
├── README.md               # Technical project documentation and overview
└── requirements.txt        # Full list of project dependencies and versions