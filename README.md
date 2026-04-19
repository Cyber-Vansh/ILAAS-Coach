# Intelligent Learning Analytics & Action System (ILAAS)

## Overview

The Intelligent Learning Analytics & Action System (ILAAS) is a machine learning-based diagnostic web application designed to identify students at risk of academic failure early in the semester. By analyzing demographic data, environmental factors, and early academic performance, educators can proactively intervene and provide necessary support to students before final exams.

## Features

- **Predictive Analytics Engine:** Utilizes a Random Forest Classifier to project early academic intervention needs.
- **Dynamic Behavioral Imputation:** Smartly infers missing behavioral traits based on the student's academic risk profile, rather than relying on unsafe assumed medians.
- **Sensitive Probability Thresholding:** Employs `predict_proba()` to combat minority class suppression, triggering early warnings if a student has even a $\ge 15\%$ probability of failing.
- **Streamlit Web Application:** A privacy-preserving, minimalist, and easy-to-use user interface (built with Streamlit) allowing teachers to input data and get real-time evaluations.
- **Statistical Insights:** Visualizations revealing the top factors contributing to student success and failure.

## Tech Stack

- **Python 3.x**
- **Machine Learning:** scikit-learn, pandas, numpy
- **Web App / UI:** Streamlit, Plotly
- **Model Serialization:** joblib

## Repository Structure

```text
ILAAS-Coach/
├── app.py                  # Main Streamlit application entry point
├── api/                    # FastAPI Backend
│   └── main.py             # FastAPI server with /predict and /agent endpoints
├── frontend/               # Next.js Application
│   ├── app/                # App router pages (Home, Assess, Coach)
│   ├── components/         # Reusable React components
│   └── package.json        # Node.js dependencies
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
├── data/                   # Raw datasets (student-mat.csv, student-por.csv)
├── data-cleaned/           # Cleaned and processed datasets
├── model/
│   ├── saved_model.pkl     # Trained Random Forest classification model (Risk)
│   ├── g3_regressor.pkl    # Trained Random Forest regressor model (G3 grade)
│   ├── retrain.py          # Script to retrain the risk classification model
│   └── training.ipynb      # Notebook detailing model training and evaluation
├── notebooks/
│   ├── data_cleaning.ipynb # Notebook for exploratory data analysis & preprocessing
│   └── agentic_study_coach.ipynb # Notebook for developing the LangGraph workflow
├── pages/
│   ├── 1_Evaluation.py     # Streamlit Evaluation Matrix page
│   ├── 2_Insights.py       # Streamlit Statistical Insights page
│   └── 3_Agentic_Coach.py  # Streamlit fallback AI Coach interface
└── ui_style.py             # Custom CSS and styling for the Streamlit UI
```

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Cyber-Vansh/ILAAS-Coach
   cd ILAAS-Coach
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - Navigate to the **Evaluation Matrix** to input a student's profile and receive an automated prediction.
   - Navigate to the **Insights** page to explore the machine learning model's feature importance and system specifications.

## Model Training

If you wish to retrain the model on new data:

1. Ensure your cleaned dataset is placed in `data-cleaned/`.
2. Run the retraining script:
   ```bash
   python model/retrain.py
   ```
3. The newly trained model will be saved as `model/saved_model.pkl` and subsequent app runs will pick up the updated model automatically.

## Dataset

The model currently uses the [Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance) (Portuguese Student Dataset) from the UCI Machine Learning Repository, focusing specifically on mathematics class data.
