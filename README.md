# 🛵 ETA & Fare Prediction — End-to-End ML App

## 📌 Problem Statement
Predict ride ETA and fare in real-time using Machine Learning,
with dynamic surge pricing based on demand conditions.

## 🚀 Live Demo
Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🤖 Model Performance
| Model | MAE | R² Score |
|-------|-----|----------|
| ETA Prediction | 1.69 minutes | 0.9950 |
| Fare Prediction | Rs. 6.85 | 0.9931 |

## ⚡ Surge Pricing Logic
| Condition | Multiplier |
|-----------|------------|
| Normal | 1.0x |
| Rush hour OR Rain | 1.5x |
| Rush hour AND Rain | 2.0x |

## 🛠️ Tech Stack
Python · Random Forest · Streamlit · Scikit-learn · Pandas · NumPy

## 📁 Files
- `app.py` — Streamlit web application
- `eta_fare_prediction.ipynb` — Full ML notebook
- `eta_model.pkl` — Trained ETA model
- `fare_model.pkl` — Trained Fare model
- `eda_eta_fare.png` — EDA visualizations
- `requirements.txt` — Dependencies
