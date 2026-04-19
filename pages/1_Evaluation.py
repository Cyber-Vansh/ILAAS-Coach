import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ui_style

st.set_page_config(page_title="Evaluation Matrix — ILAAS", page_icon="📊", layout="wide")
ui_style.apply_german_ui()

if 'step' not in st.session_state:
    st.session_state.step = 1

st.markdown("<h1>Evaluation Matrix</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='lead'>Complete the 3-step diagnostic form. "
    "The ML model will classify the student as At-Risk, Average, or High Performer.</p>",
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    try:
        m = joblib.load("model/saved_model.pkl")
        return m, m.feature_names_in_
    except Exception:
        return None, None

model, expected_columns = load_model()

if not model:
    st.error("⚠️ Model file not found. Please run `model/retrain.py` first.")
    st.stop()

current_step = min(st.session_state.step, 4)
st.markdown(f"<div class='wizard-step'>Step {current_step} of 3 — ", unsafe_allow_html=True)

step_names = {1: "Academic Grades", 2: "Student Details", 3: "Study Environment", 4: "Results"}
st.markdown(f"<div class='wizard-step'>{step_names.get(current_step, '')}</div>", unsafe_allow_html=True)

st.progress(min(current_step / 3.0, 1.0))
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.step == 1:
    with st.container():
        st.markdown("<h3>📝 Academic Performance</h3>", unsafe_allow_html=True)

        G1 = st.slider("Semester 1 Grade (0–20)", 0, 20, st.session_state.get('G1', 10),
                       help="First period grade out of 20")
        G2 = st.slider("Semester 2 Grade (0–20)", 0, 20, st.session_state.get('G2', 10),
                       help="Second period grade out of 20")

        col1, col2 = st.columns(2)
        with col1:
            failures = st.number_input("Previous Class Failures", min_value=0, max_value=4,
                                       value=st.session_state.get('failures', 0))
        with col2:
            absences = st.number_input("Days Absent This Year", min_value=0, max_value=100,
                                       value=st.session_state.get('absences', 2))

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next: Student Details →", type="primary"):
            st.session_state.G1 = G1
            st.session_state.G2 = G2
            st.session_state.failures = failures
            st.session_state.absences = absences
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    with st.container():
        st.markdown("<h3>👤 Student Profile</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Student Age", min_value=14, max_value=25,
                                  value=st.session_state.get('age', 16))
            sex = st.selectbox("Gender", ["Female", "Male"],
                               index=["Female", "Male"].index(st.session_state.get('sex', "Female")))
            address = st.selectbox("Where do they live?", ["Urban (City)", "Rural (Country)"],
                                   index=["Urban (City)", "Rural (Country)"].index(
                                       st.session_state.get('address', "Urban (City)")))

        with col2:
            school_choice = st.selectbox("Which school?",
                                         ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"],
                                         index=["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"].index(
                                             st.session_state.get('school_choice', "Gabriel Pereira (GP)")))
            reason = st.selectbox("Why did they pick this school?",
                                  ["Close to home", "Good school reputation",
                                   "Specific courses offered", "Other reason"],
                                  index=["Close to home", "Good school reputation",
                                         "Specific courses offered", "Other reason"].index(
                                      st.session_state.get('reason', "Close to home")))
            higher = st.selectbox("Do they plan to go to college?", ["Yes", "No"],
                                  index=["Yes", "No"].index(st.session_state.get('higher', "Yes")))

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                st.session_state.age = age
                st.session_state.sex = sex
                st.session_state.address = address
                st.session_state.school_choice = school_choice
                st.session_state.reason = reason
                st.session_state.higher = higher
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Next: Environment →", type="primary"):
                st.session_state.age = age
                st.session_state.sex = sex
                st.session_state.address = address
                st.session_state.school_choice = school_choice
                st.session_state.reason = reason
                st.session_state.higher = higher
                st.session_state.step = 3
                st.rerun()

elif st.session_state.step == 3:
    with st.container():
        st.markdown("<h3>🏡 Study Environment</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            studytime = st.selectbox("Weekly study hours at home?",
                                     ["Less than 2 hours", "2 to 5 hours",
                                      "5 to 10 hours", "More than 10 hours"],
                                     index=["Less than 2 hours", "2 to 5 hours",
                                            "5 to 10 hours", "More than 10 hours"].index(
                                         st.session_state.get('studytime', "2 to 5 hours")))
            famsup = st.selectbox("Family helps with studying?", ["Yes", "No"],
                                  index=["Yes", "No"].index(st.session_state.get('famsup', "Yes")))
            schoolsup = st.selectbox("Extra school support?", ["No", "Yes"],
                                     index=["No", "Yes"].index(st.session_state.get('schoolsup', "No")))

        with col2:
            paid = st.selectbox("Paid private tutoring?", ["No", "Yes"],
                                index=["No", "Yes"].index(st.session_state.get('paid', "No")))
            internet = st.selectbox("Internet access at home?", ["Yes", "No"],
                                    index=["Yes", "No"].index(st.session_state.get('internet', "Yes")))
            traveltime = st.selectbox("Commute time to school?",
                                      ["Less than 15 mins", "15 to 30 mins",
                                       "30 mins to 1 hour", "More than 1 hour"],
                                      index=["Less than 15 mins", "15 to 30 mins",
                                             "30 mins to 1 hour", "More than 1 hour"].index(
                                          st.session_state.get('traveltime', "Less than 15 mins")))

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                st.session_state.studytime = studytime
                st.session_state.famsup = famsup
                st.session_state.schoolsup = schoolsup
                st.session_state.paid = paid
                st.session_state.internet = internet
                st.session_state.traveltime = traveltime
                st.session_state.step = 2
                st.rerun()
        with col_next:
            if st.button("🔍 Run Prediction", type="primary"):
                st.session_state.studytime = studytime
                st.session_state.famsup = famsup
                st.session_state.schoolsup = schoolsup
                st.session_state.paid = paid
                st.session_state.internet = internet
                st.session_state.traveltime = traveltime
                st.session_state.step = 4
                st.rerun()

elif st.session_state.step == 4:
    st.markdown("<h3>📈 Prediction Results</h3>", unsafe_allow_html=True)

    with st.spinner("Running the model..."):
        is_high_risk = (
            st.session_state.failures > 0 or
            st.session_state.absences > 15 or
            (st.session_state.G1 + st.session_state.G2) < 20
        )

        input_data = {
            'school': 'GP' if 'GP' in st.session_state.school_choice else 'MS',
            'sex': 'M' if st.session_state.sex == 'Male' else 'F',
            'age': st.session_state.age,
            'address': 'U' if 'Urban' in st.session_state.address else 'R',
            'famsize': 'GT3', 'Pstatus': 'T', 'Medu': 2, 'Fedu': 2,
            'Mjob': 'other', 'Fjob': 'other',
            'reason': (
                'home' if 'Close' in st.session_state.reason else
                'reputation' if 'reputation' in st.session_state.reason else
                'course' if 'courses' in st.session_state.reason else 'other'
            ),
            'guardian': 'mother',
            'traveltime': (1 if '15' in st.session_state.traveltime else
                           2 if '30' in st.session_state.traveltime else
                           3 if '1 hour' in st.session_state.traveltime else 4),
            'studytime': (1 if 'Less' in st.session_state.studytime else
                          2 if '2 to 5' in st.session_state.studytime else
                          3 if '5 to 10' in st.session_state.studytime else 4),
            'failures': st.session_state.failures,
            'schoolsup': 'yes' if st.session_state.schoolsup == "Yes" else 'no',
            'famsup': 'yes' if st.session_state.famsup == "Yes" else 'no',
            'paid': 'yes' if st.session_state.paid == "Yes" else 'no',
            'activities': 'no' if is_high_risk else 'yes',
            'nursery': 'yes',
            'higher': 'yes' if st.session_state.higher == "Yes" else 'no',
            'internet': 'yes' if st.session_state.internet == "Yes" else 'no',
            'romantic': 'yes' if is_high_risk else 'no',
            'famrel': 2 if is_high_risk else 4,
            'freetime': 5 if is_high_risk else 3,
            'goout': 5 if is_high_risk else 3,
            'health': 4,
            'Dalc': 3 if is_high_risk else 1,
            'Walc': 4 if is_high_risk else 1,
            'absences': st.session_state.absences,
            'G1': st.session_state.G1,
            'G2': st.session_state.G2
        }

        df_raw = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_raw)

        final_features = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in expected_columns:
            if col in df_encoded.columns:
                final_features[col] = df_encoded[col]

        numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
        for col in final_features.columns:
            if col in numeric_cols:
                final_features[col] = pd.to_numeric(final_features[col], downcast='integer')
            else:
                final_features[col] = final_features[col].astype('bool')

        classes = model.classes_
        probs = model.predict_proba(final_features)[0]
        prob_map = dict(zip(classes, probs))

        if prob_map.get('At Risk', 0) >= 0.15:
            prediction = 'At Risk'
        elif prob_map.get('High Performer', 0) > prob_map.get('Average', 0):
            prediction = 'High Performer'
        else:
            prediction = 'Average'

    if prediction == "At Risk":
        st.markdown(f"""
        <div class="result-block risk-high">
            <div class="result-title">⚠️ EVALUATION: High Priority Intervention Required</div>
            <div class="result-desc">
                Classification: <strong>AT RISK</strong><br><br>
                The model detects significant deviation from the success baseline.
                Immediate academic support is recommended. Consider using the 
                <strong>AI Study Coach</strong> to generate a personalized recovery plan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == "Average":
        st.markdown(f"""
        <div class="result-block risk-avg">
            <div class="result-title">📊 EVALUATION: Standard Monitoring Recommended</div>
            <div class="result-desc">
                Classification: <strong>AVERAGE</strong><br><br>
                The student aligns with the standard performance baseline.
                Targeted improvement strategies could elevate their trajectory.
                Try the <strong>AI Study Coach</strong> for a tailored study plan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-block risk-low">
            <div class="result-title">✅ EVALUATION: Optimal Trajectory Confirmed</div>
            <div class="result-desc">
                Classification: <strong>HIGH PERFORMER</strong><br><br>
                Data vectors align strongly with highest historical success patterns.
                No structural intervention required. The <strong>AI Study Coach</strong> 
                can still help set stretch goals and advanced resources.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Probability Breakdown</h3>", unsafe_allow_html=True)
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    for col, (cls, color) in zip(
        [prob_col1, prob_col2, prob_col3],
        [("At Risk", "#f87171"), ("Average", "#fbbf24"), ("High Performer", "#34d399")]
    ):
        prob_val = prob_map.get(cls, 0)
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:{color}; -webkit-text-fill-color:{color};">
                    {prob_val:.0%}
                </div>
                <div class="stat-label">{cls}</div>
            </div>
            """, unsafe_allow_html=True)

    st.session_state['last_prediction'] = prediction
    st.session_state['last_prob_map'] = prob_map

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_new = st.columns(2)
    with col_back:
        if st.button("← Modify Inputs"):
            st.session_state.step = 3
            st.rerun()
    with col_new:
        if st.button("🔄 Start New Evaluation", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
