import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from src.ml_pipeline import recommendation_for_label


st.set_page_config(
    page_title="ILAAS-COACH",
    page_icon="STR",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        :root {
            --bg-primary: #050816;
            --bg-elevated: rgba(11, 15, 25, 0.96);
            --bg-elevated-soft: rgba(13, 19, 33, 0.9);
            --grid-line: rgba(148, 163, 184, 0.18);
            --accent-cyan: #22d3ee;
            --accent-pink: #f472b6;
            --accent-purple: #a855f7;
            --accent-red: #fb7185;
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
        }

        * {
            font-family: 'Inter', -apple-system, sans-serif !important;
        }

        body {
            background:
                linear-gradient(135deg, rgba(34,211,238,0.06), transparent 45%),
                linear-gradient(225deg, rgba(244,114,182,0.10), transparent 40%),
                radial-gradient(circle at top left, rgba(56,189,248,0.18), transparent 55%),
                radial-gradient(circle at bottom right, rgba(244,114,182,0.22), transparent 60%),
                var(--bg-primary) !important;
            color: var(--text-main);
        }

        .main-header {
            font-size: 4rem;
            font-weight: 800;
            letter-spacing: -2px;
            color: var(--text-main);
            text-align: left;
            margin-bottom: 3rem;
            line-height: 1;
            text-transform: uppercase;
            position: relative;
        }

        .main-header::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -12px;
            width: 140px;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
            box-shadow: 0 0 20px rgba(244,114,182,0.6);
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(24,24,27,0.92));
            padding: 2rem;
            border: 1px solid var(--grid-line);
            border-top: 3px solid var(--accent-purple);
            color: var(--text-main);
            margin: 0.5rem 0;
            border-radius: 0px;
            box-shadow: 0 22px 36px rgba(15,23,42,0.65);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: 0.22;
            background-image: linear-gradient(var(--grid-line) 1px, transparent 1px),
                              linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
            background-size: 26px 26px;
            mix-blend-mode: soft-light;
            pointer-events: none;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1;
        }

        .metric-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--text-muted);
        }

        .sidebar-section {
            background: transparent;
            padding: 0.5rem 0;
            border: none;
            border-top: 1px solid rgba(148,163,184,0.35);
            margin: 0.75rem 0;
            border-radius: 0;
            box-shadow: none;
        }

        .stButton > button {
            background: radial-gradient(circle at top left, var(--accent-cyan), var(--accent-purple));
            color: #0b1020;
            border: 1px solid rgba(148,163,184,0.4);
            padding: 0.75rem 1.5rem;
            border-radius: 0px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .stButton > button::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 10% 0%, rgba(255,255,255,0.38), transparent 55%);
            opacity: 0;
            transition: opacity 160ms ease-out;
        }

        .stButton > button:hover::after {
            opacity: 1;
        }

        .stButton > button:hover {
            box-shadow: 0 0 24px rgba(34,211,238,0.75);
        }

        .plot-container {
            background: var(--bg-elevated);
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--grid-line);
            margin: 1rem 0;
            position: relative;
        }

        .status-indicator {
            display: inline-block;
            width: 18px;
            height: 4px;
            margin-right: 8px;
            box-shadow: 0 0 16px rgba(248,250,252,0.95);
        }

        .status-high { background: linear-gradient(90deg, var(--accent-red), #fecaca); }
        .status-medium { background: linear-gradient(90deg, var(--accent-purple), #e5deff); }
        .status-low { background: linear-gradient(90deg, #4b5563, #9ca3af); }

        .info-box {
            border: 1px solid var(--grid-line);
            padding: 1.5rem;
            margin: 1rem 0;
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.88));
        }

        .warning-box {
            background: radial-gradient(circle at top left, rgba(248,113,113,0.28), rgba(127,29,29,0.95));
            color: #fee2e2;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(248,113,113,0.65);
            box-shadow: 0 0 22px rgba(248,113,113,0.75);
        }

        .success-box {
            border: 1px solid var(--grid-line);
            padding: 1.5rem;
            margin: 1rem 0;
            background: linear-gradient(135deg, rgba(5,150,105,0.18), rgba(15,23,42,0.94));
        }

        /* Layout Grid */
        .swiss-grid {
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 1.1rem;
        }

        .swiss-cell-span-3 { grid-column: span 3 / span 3; }
        .swiss-cell-span-4 { grid-column: span 4 / span 4; }
        .swiss-cell-span-6 { grid-column: span 6 / span 6; }
        .swiss-cell-span-12 { grid-column: span 12 / span 12; }

        @media (max-width: 900px) {
            .main-header {
                font-size: 2.5rem;
            }
            .swiss-grid {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }
            .swiss-cell-span-3,
            .swiss-cell-span-4,
            .swiss-cell-span-6,
            .swiss-cell-span-12 {
                grid-column: 1 / -1;
            }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    df = pd.read_csv("data-cleaned/processed_student_mat.csv")
    X = df.drop("risk_level", axis=1)
    X_dummies = pd.get_dummies(X)
    model = joblib.load("src/student_risk_model.pkl")
    scaler = joblib.load("src/scaler.pkl")
    return X, X_dummies.columns.tolist(), model, scaler


def sidebar():
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("ILAAS COACH")
    st.sidebar.markdown("Individual Student Risk Analysis")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.write("Model trained on cleaned historical data from the course dataset.")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)


def student_input_form():
    st.markdown('<h1 class="main-header">» STUDENT RISK PREDICTOR</h1>', unsafe_allow_html=True)
    X_base, feature_columns, model, scaler = load_artifacts()

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=15, max_value=22, value=16)
            Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
            Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
            traveltime = st.slider("Travel time to school (1-4)", 1, 4, 1)
            studytime = st.slider("Weekly study time (1-4)", 1, 4, 2)
        with col2:
            failures = st.slider("Number of past class failures", 0, 4, 0)
            famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
            freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
            goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
            Dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
        with col3:
            Walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 1)
            health = st.slider("Current health status (1-5)", 1, 5, 3)
            absences = st.number_input("Number of absences", min_value=0, max_value=93, value=4)
            G1 = st.number_input("First period grade G1 (0-20)", min_value=0, max_value=20, value=10)
            G2 = st.number_input("Second period grade G2 (0-20)", min_value=0, max_value=20, value=10)

        st.markdown('<div class="metric-label">Demographics and support</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            school_MS = st.checkbox("School is MS (else GP)", value=False)
            sex_M = st.checkbox("Student is male", value=False)
            address_U = st.checkbox("Urban address", value=True)
            famsize_LE3 = st.checkbox("Family size ≤ 3", value=False)
            Pstatus_T = st.checkbox("Parents living together", value=True)
        with col5:
            Mjob_health = st.checkbox("Mother works in health", value=False)
            Mjob_other = st.checkbox("Mother job: other", value=True)
            Mjob_services = st.checkbox("Mother in services", value=False)
            Mjob_teacher = st.checkbox("Mother is teacher", value=False)
            Fjob_health = st.checkbox("Father works in health", value=False)
        with col6:
            Fjob_other = st.checkbox("Father job: other", value=True)
            Fjob_services = st.checkbox("Father in services", value=False)
            Fjob_teacher = st.checkbox("Father is teacher", value=False)
            romantic_yes = st.checkbox("In a romantic relationship", value=False)

        st.markdown('<div class="metric-label">Motivation and resources</div>', unsafe_allow_html=True)
        col7, col8, col9 = st.columns(3)
        with col7:
            reason_home = st.checkbox("Reason: close to home", value=False)
            reason_other = st.checkbox("Reason: other", value=False)
            reason_reputation = st.checkbox("Reason: school reputation", value=True)
        with col8:
            guardian_mother = st.checkbox("Guardian is mother", value=True)
            guardian_other = st.checkbox("Guardian is other", value=False)
            schoolsup_yes = st.checkbox("School support", value=False)
            famsup_yes = st.checkbox("Family support", value=True)
        with col9:
            paid_yes = st.checkbox("Extra paid classes", value=False)
            activities_yes = st.checkbox("Extracurricular activities", value=True)
            nursery_yes = st.checkbox("Attended nursery school", value=True)
            higher_yes = st.checkbox("Wants higher education", value=True)
            internet_yes = st.checkbox("Internet access at home", value=True)

        submitted = st.form_submit_button("Analyze student")

    if not submitted:
        return

    new_student = {
        "age": age,
        "Medu": Medu,
        "Fedu": Fedu,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
        "school_MS": school_MS,
        "sex_M": sex_M,
        "address_U": address_U,
        "famsize_LE3": famsize_LE3,
        "Pstatus_T": Pstatus_T,
        "Mjob_health": Mjob_health,
        "Mjob_other": Mjob_other,
        "Mjob_services": Mjob_services,
        "Mjob_teacher": Mjob_teacher,
        "Fjob_health": Fjob_health,
        "Fjob_other": Fjob_other,
        "Fjob_services": Fjob_services,
        "Fjob_teacher": Fjob_teacher,
        "reason_home": reason_home,
        "reason_other": reason_other,
        "reason_reputation": reason_reputation,
        "guardian_mother": guardian_mother,
        "guardian_other": guardian_other,
        "schoolsup_yes": schoolsup_yes,
        "famsup_yes": famsup_yes,
        "paid_yes": paid_yes,
        "activities_yes": activities_yes,
        "nursery_yes": nursery_yes,
        "higher_yes": higher_yes,
        "internet_yes": internet_yes,
        "romantic_yes": romantic_yes,
    }

    base_columns = X_base.columns.tolist()
    new_df_base = pd.DataFrame([[new_student.get(col, X_base[col].iloc[0]) for col in base_columns]], columns=base_columns)
    new_dummies = pd.get_dummies(new_df_base)
    new_dummies = new_dummies.reindex(columns=feature_columns, fill_value=0)
    new_scaled = scaler.transform(new_dummies.values)

    probabilities = model.predict_proba(new_scaled)[0]
    predicted_label = model.predict(new_scaled)[0]

    st.markdown('<div class="metric-label">Predicted risk category</div>', unsafe_allow_html=True)
    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>MODEL PREDICTION</div>"
            f"<div class='metric-value'>{predicted_label}</div></div>",
            unsafe_allow_html=True,
        )
    with col_side:
        labels = model.classes_
        prob_df = pd.DataFrame({"label": labels, "probability": probabilities})
        fig = px.bar(prob_df, x="label", y="probability", color="label", range_y=[0, 1])
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    recommendation = recommendation_for_label(predicted_label)
    st.subheader("Study recommendation")
    st.markdown(
        f"""
        <div class="info-box">
            <p>{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_css()
    sidebar()
    student_input_form()
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "© 2024 ILAAS Coach - Individual Risk Prediction using Trained Model"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
