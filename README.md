# Intelligent Learning Analytics & Action System (ILAAS)

## Overview

The Intelligent Learning Analytics & Action System (ILAAS) is a machine learning-based diagnostic web application designed to identify students at risk of academic failure early in the semester. By analyzing demographic data, environmental factors, and early academic performance, educators can proactively intervene and provide necessary support to students before final exams.

## Features

- **Predictive Analytics Engine:** Utilizes a Random Forest Classifier to project early academic intervention needs.
- **Dynamic Behavioral Imputation:** Smartly infers missing behavioral traits based on the student's academic risk profile, rather than relying on unsafe assumed medians.
- **Sensitive Probability Thresholding:** Employs `predict_proba()` to combat minority class suppression, triggering early warnings if a student has even a $\ge 15\%$ probability of failing.
- **Next.js Web Application:** A modern, responsive web interface (built with Next.js and React) allowing teachers to input data and get real-time evaluations.
- **Statistical Insights:** Visualizations revealing the top factors contributing to student success and failure.

## Tech Stack

- **Frontend:** Next.js, React, TypeScript, CSS Modules
- **Backend:** Python FastAPI
- **Machine Learning:** scikit-learn, pandas, numpy
- **Data Visualization:** Plotly
- **Model Serialization:** joblib

## Repository Structure

```text
ILAAS-Coach/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Root layout component
│   ├── page.tsx            # Home page
│   ├── assess/             # Student assessment page
│   │   └── page.tsx
│   └── coach/              # AI Coach page
│       └── page.tsx
├── api/                    # Python FastAPI Backend
│   ├── index.py            # FastAPI server with /predict and /agent endpoints
│   └── model/
│       ├── retrain.py      # Script to retrain models
│       └── training.ipynb  # Notebook for model training and evaluation
├── components/             # Reusable React components
│   └── Nav.tsx             # Navigation component
├── data/                   # Raw datasets
│   ├── student-mat.csv
│   └── student-por.csv
├── data-cleaned/           # Cleaned and processed datasets
├── model/                  # Trained ML models
├── notebooks/              # Jupyter notebooks for analysis
│   ├── data_cleaning.ipynb
│   └── agentic_study_coach.ipynb
├── utils/                  # Utility functions and API client
│   └── api.ts
├── package.json            # Node.js dependencies
├── requirements.txt        # Python dependencies
├── next.config.ts          # Next.js configuration
├── tsconfig.json           # TypeScript configuration
├── README.md               # Project documentation
└── .gitignore              # Git ignored files
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

3. **Install Node.js dependencies:**

   ```bash
   npm install
   ```

4. **Install Python dependencies (for backend):**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Next.js development server:**

   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:3000`.

2. **Interact with the App:**
   - Navigate to the **Assess** page to input a student's profile and receive an automated prediction.
   - Navigate to the **Coach** page to interact with the AI-powered study coach for personalized recommendations.
   - View the home page for an overview of the system.

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
