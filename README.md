# 🏥 LifePulse - AI-Powered Health Analytics Platform

[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://lifepulse-9vz4.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **AI-driven health prediction and monitoring platform** with machine learning models for heart disease, sleep disorders, migraine assessment, and personalized health scoring.

🔗 **Live Application:** [https://lifepulse-9vz4.onrender.com/](https://lifepulse-9vz4.onrender.com/)

---

## 📋 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Models & Accuracy](#-ml-models--accuracy)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🧮 Health Score Calculator
- Comprehensive health assessment based on vital metrics
- BMI, blood pressure, cholesterol analysis
- Lifestyle factors evaluation (exercise, sleep, smoking)
- **Accuracy:** 54.8% R² score

### ❤️ Heart Disease Prediction
- Advanced cardiovascular risk assessment
- 22-parameter analysis including BMI, cholesterol, diabetes, lifestyle
- Binary classification (Disease/No Disease)
- **Accuracy:** 91%

### 😴 Sleep Disorder Detection
- Identifies sleep apnea, insomnia, and normal sleep patterns
- Analyzes sleep duration, quality, physical activity, stress levels
- Multi-class classification
- **Accuracy:** 87.1%

### 🤕 Migraine Risk Assessment
- Predicts migraine susceptibility
- Evaluates triggers, frequency, lifestyle factors
- Binary classification model
- **Accuracy:** 51%

### 🥗 Nutrition Tracker (Powered by Gemini AI)
- AI-driven dietary analysis
- Personalized nutrition recommendations
- USDA FoodData Central integration
- Real-time nutritional insights

---

## 🛠️ Tech Stack

### Backend
- **Framework:** Flask 2.3.2
- **WSGI Server:** Gunicorn
- **ML Libraries:** scikit-learn, pandas, numpy, joblib
- **AI Integration:** Google Gemini API

### Frontend
- **UI Framework:** Bootstrap 5.3.0
- **Icons:** Bootstrap Icons 1.11.1
- **Animations:** AOS (Animate On Scroll) 2.3.1
- **Custom CSS:** Responsive mobile-first design

### Deployment
- **Platform:** Render
- **Version Control:** Git with Git LFS (for model files)
- **Python Version:** 3.13

---

## 🤖 ML Models & Accuracy

| Feature | Algorithm | Accuracy | Model Size | Parameters |
|---------|-----------|----------|------------|------------|
| Heart Disease | Random Forest | **91%** | 6.2 MB | 22 features |
| Sleep Disorder | Random Forest | **87.1%** | Multi-file | 13 features |
| Health Score | Random Forest Regressor | **54.8% R²** | 3 models | 11 features |
| Migraine | SVM | **51%** | < 1 MB | 8 features |

### Model Details

**Heart Disease Model:**
- Features: BMI, Age, Sex, High BP, High Cholesterol, Diabetes, Physical Activity, etc.
- Training: BRFSS 2015 dataset
- Output: Binary (0 = No Disease, 1 = Disease)

**Sleep Disorder Model:**
- Features: Sleep Duration, Quality, Physical Activity, Stress, BMI, Heart Rate, Daily Steps
- Output: Multi-class (Normal, Sleep Apnea, Insomnia)
- Preprocessing: Label encoding, standard scaling

**Health Score Model:**
- Features: Age, BMI, Blood Pressure, Cholesterol, Exercise, Sleep Hours, Smoking
- Output: Continuous health score (0-100)
- Ensemble: Random Forest with feature importance

**Migraine Model:**
- Features: Age, Frequency, Duration, Triggers, Lifestyle factors
- Output: Binary (0 = Low Risk, 1 = High Risk)

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager
- Git with Git LFS

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/rogerdemello/LifePulse.git
cd LifePulse
```

2. **Install Git LFS** (for model files):
```bash
git lfs install
git lfs pull
```

3. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key
USDA_API_KEY=your-usda-api-key
```

6. **Run the application:**
```bash
# Development mode
python run.py

# Production mode (with Gunicorn)
gunicorn wsgi:app
```

7. **Access the application:**
```
http://localhost:5000
```

---

## 📖 Usage

### Health Score Calculator
1. Navigate to `/health`
2. Enter vital metrics (age, weight, height, blood pressure, etc.)
3. Submit form to receive comprehensive health score
4. View personalized recommendations

### Heart Disease Prediction
1. Go to `/heart_disease`
2. Fill in 22-parameter health questionnaire
3. Get instant cardiovascular risk assessment
4. Receive prevention suggestions

### Sleep Disorder Detection
1. Visit `/sleep`
2. Provide sleep patterns and lifestyle data
3. Receive disorder classification (Normal/Apnea/Insomnia)
4. Get sleep improvement tips

### Migraine Assessment
1. Access `/migraine`
2. Answer migraine-specific questions
3. Get risk prediction
4. View trigger analysis

### Nutrition Tracker
1. Navigate to `/nutrition`
2. Input food items or meal description
3. Get AI-powered nutritional analysis
4. Receive dietary recommendations

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Homepage with feature overview |
| `/health` | GET/POST | Health score calculator |
| `/heart_disease` | GET/POST | Heart disease prediction |
| `/sleep` | GET/POST | Sleep disorder detection |
| `/migraine` | GET/POST | Migraine risk assessment |
| `/health-score` | GET/POST | Detailed health scoring |
| `/nutrition` | GET/POST | Nutrition analysis with AI |

---

## 📁 Project Structure

```
LifePulse/
├── app/
│   ├── __init__.py              # Flask app initialization
│   ├── app.py                   # Application factory
│   ├── models/                  # ML model files (.pkl)
│   │   ├── heart/               # Heart disease models
│   │   ├── sleep/               # Sleep disorder models
│   │   ├── health_score/        # Health score models
│   │   └── *.pkl                # Migraine models
│   ├── routes/                  # Blueprint routes
│   │   ├── calculator_routes.py
│   │   ├── heart.py
│   │   ├── sleep.py
│   │   ├── nutrition.py
│   │   └── health_score.py
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JS, images
│   └── utils/                   # Helper functions
│       ├── predictor.py         # ML prediction logic
│       ├── model_loader.py      # Model management
│       ├── gemini.py            # AI integration
│       └── nutrition.py         # USDA API handler
├── ml_model/                    # Training scripts & datasets
│   ├── Heart_Disease.py
│   ├── sleep_dis.py
│   ├── Health_score.py
│   └── data/
├── Procfile                     # Render deployment config
├── wsgi.py                      # WSGI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🌍 Deployment

### Render Deployment (Recommended)

1. **Fork/Clone this repository**

2. **Create new Web Service on Render:**
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
   - Environment: Python 3

3. **Set Environment Variables:**
   - `SECRET_KEY`: Flask secret key
   - `GEMINI_API_KEY`: Google Gemini API key
   - `USDA_API_KEY`: USDA FoodData Central API key

4. **Enable Git LFS** in Render dashboard

5. **Deploy!** 🚀

### Local Deployment
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

---

## 🧪 Model Training

To retrain models with updated datasets:

```bash
# Activate virtual environment
source venv/bin/activate

# Train heart disease model
python ml_model/Heart_Disease.py

# Train sleep disorder model
python ml_model/sleep_dis.py

# Train health score model
python ml_model/Health_score.py

# Train migraine model
python ml_model/Migrain.py
```

Models are saved with timestamps in `app/models/` directory.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Roger Demello**
- GitHub: [@rogerdemello](https://github.com/rogerdemello)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/roger-demello)
- Live Demo: [LifePulse](https://lifepulse-9vz4.onrender.com/)

---

## 🙏 Acknowledgments

- BRFSS 2015 dataset for heart disease model
- Sleep Health & Lifestyle dataset
- USDA FoodData Central for nutrition data
- Google Gemini AI for intelligent recommendations
- Bootstrap team for UI components
- AOS library for smooth animations

---

## 📊 Stats

![GitHub repo size](https://img.shields.io/github/repo-size/rogerdemello/LifePulse)
![GitHub stars](https://img.shields.io/github/stars/rogerdemello/LifePulse?style=social)
![GitHub forks](https://img.shields.io/github/forks/rogerdemello/LifePulse?style=social)

---
