# 🧠 NeuroAI: Clinical Decision Support System

NeuroAI is a full-stack, AI-powered web application designed to assist radiologists and neuro-oncologists in analyzing brain MRI scans. It utilizes a Hybrid U-Net deep learning model for high-fidelity tumor segmentation, providing volumetric calculations, WHO staging estimations, and automated clinical reporting.

## 🚀 Features
* **AI Tumor Segmentation:** High-accuracy automated boundary detection using a Hybrid U-Net model.
* **Volumetric Analysis:** Spherical approximation algorithms to estimate 3D tumor volume and max diameter.
* **Clinical Pathway Generation:** Dynamic treatment recommendations based on predicted WHO grading.
* **Professional Export:** Generates clinic-ready, formatted PDF reports for Electronic Health Records (EHR).

## 🛠️ Tech Stack
* **Frontend:** React, TypeScript, Vite, Tailwind CSS, Lucide Icons.
* **Backend:** FastAPI, Python, Uvicorn.
* **Machine Learning:** TensorFlow, Keras, OpenCV, Scikit-Image.

## ⚙️ Local Setup Instructions

### 1. Backend Setup
Navigate to the root directory and set up the Python environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
cd backend
pip install -r requirements.txt
python main.py
```
*The backend will run on http://localhost:8000*

### 2. Frontend Setup
Open a new terminal and navigate to the frontend directory:
```
cd frontend
npm install
npm run dev
```
*The frontend will run on http://localhost:5173*
