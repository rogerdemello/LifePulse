# LifePulse Project Overview

## Project Description
LifePulse is a health analytics platform that leverages machine learning models to predict and analyze various health conditions. The application provides users with insights into heart disease risk, sleep disorders, migraine likelihood, and overall health score based on user input and historical health data.

## Features
- **Heart Disease Prediction:** Uses a Random Forest model to estimate the risk of heart disease.
- **Sleep Disorder Detection:** Predicts sleep disorders using user lifestyle and biometric data.
- **Migraine Prediction:** Estimates migraine risk using SVM-based classification.
- **Health Score Calculation:** Provides a composite health score using regression models.
- **Interactive Web Interface:** User-friendly forms and result pages for health predictions.
- **Data Visualization:** Charts and graphs for health metrics and predictions.

## Machine Learning Models
| Feature         | Algorithm                | Accuracy   | Model Size | Parameters    |
|----------------|--------------------------|------------|------------|---------------|
| Heart Disease  | Random Forest            | 91%        | 6.2 MB     | 22 features   |
| Sleep Disorder | Random Forest            | 87.1%      | Multi-file | 13 features   |
| Health Score   | Random Forest Regressor  | 54.8% R²   | 3 models   | 11 features   |
| Migraine       | SVM                      | 51%        | < 1 MB     | 8 features    |

## Project Structure
```
LifePulse/
├── app/
│   ├── models/
│   ├── routes/
│   ├── static/
│   ├── templates/
│   └── utils/
├── ml_model/
│   ├── Health_score.py
│   ├── Heart_Disease.py
│   ├── Migrain.py
│   ├── sleep_dis.py
│   └── data/
├── requirements.txt
├── run.py
├── wsgi.py
└── Procfile
```

## How It Works
1. **User Input:** Users fill out health forms with personal and biometric data.
2. **Prediction:** Data is processed and passed to the relevant ML model.
3. **Result Display:** Predictions and scores are shown on result pages with visualizations.
4. **Model Training:** Models are trained on curated health datasets and can be retrained for improvements.

## Technologies Used
- Python (Flask)
- Scikit-learn
- HTML/CSS/JavaScript
- Jupyter Notebooks (for model development)

## Advancements & Future Work
- Integrate more health conditions and predictive models.
- Improve model accuracy with larger and more diverse datasets.
- Add user authentication and personalized dashboards.
- Implement real-time health tracking via wearable device integration.
- Enhance data visualization and reporting features.
- Deploy as a scalable cloud service.

## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python run.py`
4. Access the web interface at `http://localhost:5000`

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License
This project is licensed under the MIT License.
