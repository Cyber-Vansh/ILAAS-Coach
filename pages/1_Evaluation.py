import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ui_style

st.set_page_config(page_title="Evaluation Matrix", layout="wide")
ui_style.apply_german_ui()

if 'step' not in st.session_state:
    st.session_state.step = 1

col_main, _ = st.columns([1, 0.05])

with col_main:
    st.markdown("<h1>Evaluation Matrix.</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Complete the multi-step diagnostic assessment to project early academic intervention needs.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        try:
            m = joblib.load("model/saved_model.pkl")
            return m, m.feature_names_in_
        except Exception:
            return None, None

    model, expected_columns = load_model()

    if not model:
        st.error("System Error: Analytics model unavailable.")
        st.stop()
        
    st.markdown(f"<div class='wizard-step'>Step {min(st.session_state.step, 4)} of 4</div>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 4.0)
    
    form_container = st.container()
    
    with form_container:
        if st.session_state.step == 1:
            st.markdown("<h3>01. Academic Grades</h3>", unsafe_allow_html=True)
            G1 = st.slider("Semester 1 Grade (0-20)", 0, 20, st.session_state.get('G1', 10))
            G2 = st.slider("Semester 2 Grade (0-20)", 0, 20, st.session_state.get('G2', 10))
            failures = st.number_input("How many classes have they failed before?", min_value=0, max_value=4, value=st.session_state.get('failures', 0))
            absences = st.number_input("How many days have they been absent?", min_value=0, max_value=100, value=st.session_state.get('absences', 2))
            
            if st.button("Next: Demographics"):
                st.session_state.G1 = G1
                st.session_state.G2 = G2
                st.session_state.failures = failures
                st.session_state.absences = absences
                st.session_state.step = 2
                st.rerun()
            
        elif st.session_state.step == 2:
            st.markdown("<h3>02. Student Basics</h3>", unsafe_allow_html=True)
            age = st.number_input("Student Age", min_value=14, max_value=25, value=st.session_state.get('age', 16))
            
            sex_options = ["Female", "Male"]
            default_sex = sex_options.index(st.session_state.get('sex', "Female"))
            sex = st.selectbox("Gender", sex_options, index=default_sex)
            
            addr_options = ["Urban (City)", "Rural (Country)"]
            default_addr = addr_options.index(st.session_state.get('address', "Urban (City)"))
            address = st.selectbox("Where do they live?", addr_options, index=default_addr)
            
            inst_options = ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"]
            default_inst = inst_options.index(st.session_state.get('school_choice', "Gabriel Pereira (GP)"))
            school_choice = st.selectbox("Which school do they attend?", inst_options, index=default_inst)
            
            reason_options = ["Close to home", "Good school reputation", "Specific courses offered", "Other reason"]
            default_reason = reason_options.index(st.session_state.get('reason', "Close to home"))
            reason = st.selectbox("Why did they choose this school?", reason_options, index=default_reason)
            
            higher_options = ["Yes", "No"]
            default_higher = higher_options.index(st.session_state.get('higher', "Yes"))
            higher = st.selectbox("Do they want to go to college/university?", higher_options, index=default_higher)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back"):
                    st.session_state.age = age
                    st.session_state.sex = sex
                    st.session_state.address = address
                    st.session_state.school_choice = school_choice
                    st.session_state.reason = reason
                    st.session_state.higher = higher
                    st.session_state.step = 1
                    st.rerun()
            with col2:
                if st.button("Next: Environment"):
                    st.session_state.age = age
                    st.session_state.sex = sex
                    st.session_state.address = address
                    st.session_state.school_choice = school_choice
                    st.session_state.reason = reason
                    st.session_state.higher = higher
                    st.session_state.step = 3
                    st.rerun()

        elif st.session_state.step == 3:
            st.markdown("<h3>03. Home & Study Environment</h3>", unsafe_allow_html=True)
            
            study_opts = ["Less than 2 hours", "2 to 5 hours", "5 to 10 hours", "More than 10 hours"]
            default_study = study_opts.index(st.session_state.get('studytime', "2 to 5 hours"))
            studytime = st.selectbox("How much do they study at home per week?", study_opts, index=default_study)
            
            famsup_opts = ["Yes", "No"]
            default_famsup = famsup_opts.index(st.session_state.get('famsup', "Yes"))
            famsup = st.selectbox("Does the family help with studying?", famsup_opts, index=default_famsup)
            
            schoolsup_opts = ["No", "Yes"]
            default_schoolsup = schoolsup_opts.index(st.session_state.get('schoolsup', "No"))
            schoolsup = st.selectbox("Do they get extra educational support from the school?", schoolsup_opts, index=default_schoolsup)
            
            paid_opts = ["No", "Yes"]
            default_paid = paid_opts.index(st.session_state.get('paid', "No"))
            paid = st.selectbox("Do they have paid extra tutoring?", paid_opts, index=default_paid)
            
            net_opts = ["Yes", "No"]
            default_net = net_opts.index(st.session_state.get('internet', "Yes"))
            internet = st.selectbox("Do they have internet access at home?", net_opts, index=default_net)
            
            travel_opts = ["Less than 15 mins", "15 to 30 mins", "30 mins to 1 hour", "More than 1 hour"]
            default_travel = travel_opts.index(st.session_state.get('traveltime', "Less than 15 mins"))
            traveltime = st.selectbox("How long is their commute to school?", travel_opts, index=default_travel)
            
            st.markdown("<p style='font-size: 0.85rem; color:#888; margin-top:2rem;'>*Note: The AI uses median statistical averages for any blank personality traits.*</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back"):
                    st.session_state.studytime = studytime
                    st.session_state.famsup = famsup
                    st.session_state.schoolsup = schoolsup
                    st.session_state.paid = paid
                    st.session_state.internet = internet
                    st.session_state.traveltime = traveltime
                    st.session_state.step = 2
                    st.rerun()
            with col2:
                if st.button("Predict Student Performance", type="primary"):
                    st.session_state.studytime = studytime
                    st.session_state.famsup = famsup
                    st.session_state.schoolsup = schoolsup
                    st.session_state.paid = paid
                    st.session_state.internet = internet
                    st.session_state.traveltime = traveltime
                    st.session_state.step = 4
                    st.rerun()

        elif st.session_state.step == 4:
            st.markdown("<h3>Analysis Output</h3>", unsafe_allow_html=True)
            with st.spinner("Analyzing student data..."):
                is_high_risk_academic = st.session_state.failures > 0 or st.session_state.absences > 15 or (st.session_state.G1 + st.session_state.G2) < 20
                
                input_data = {
                    'school': 'GP' if 'GP' in st.session_state.school_choice else 'MS',
                    'sex': 'M' if st.session_state.sex == 'Male' else 'F',
                    'age': st.session_state.age,
                    'address': 'U' if 'Urban' in st.session_state.address else 'R',
                    'famsize': 'GT3', 'Pstatus': 'T', 'Medu': 2, 'Fedu': 2,
                    'Mjob': 'other', 'Fjob': 'other', 
                    'reason': 'home' if 'Close' in st.session_state.reason else ('reputation' if 'reputation' in st.session_state.reason else ('course' if 'courses' in st.session_state.reason else 'other')),
                    'guardian': 'mother',
                    'traveltime': 1 if '15' in st.session_state.traveltime else (2 if '30' in st.session_state.traveltime else (3 if '1 hour' in st.session_state.traveltime else 4)),
                    'studytime': 1 if 'Less' in st.session_state.studytime else (2 if '2 to 5' in st.session_state.studytime else (3 if '5 to 10' in st.session_state.studytime else 4)),
                    'failures': st.session_state.failures,
                    'schoolsup': 'yes' if st.session_state.schoolsup == "Yes" else 'no',
                    'famsup': 'yes' if st.session_state.famsup == "Yes" else 'no',
                    'paid': 'yes' if st.session_state.paid == "Yes" else 'no',
                    'activities': 'no' if is_high_risk_academic else 'yes', 
                    'nursery': 'yes',
                    'higher': 'yes' if "Yes" in st.session_state.higher else 'no',
                    'internet': 'yes' if "Yes" in st.session_state.internet else 'no',
                    'romantic': 'yes' if is_high_risk_academic else 'no',
                    
                    'famrel': 2 if is_high_risk_academic else 4, 
                    'freetime': 5 if is_high_risk_academic else 3, 
                    'goout': 5 if is_high_risk_academic else 3, 
                    'health': 4,
                    'Dalc': 3 if is_high_risk_academic else 1, 
                    'Walc': 4 if is_high_risk_academic else 1,
                    
                    'absences': st.session_state.absences,
                    'G1': st.session_state.G1, 'G2': st.session_state.G2
                }
                
                df_raw = pd.DataFrame([input_data])
                df_encoded = pd.get_dummies(df_raw)
                
                final_features = pd.DataFrame(0, index=[0], columns=expected_columns)
                
                for col in expected_columns:
                    if col in df_encoded.columns:
                        final_features[col] = df_encoded[col]
                
                numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
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
                    <div class="result-title">EVALUATION: High Priority Intervention Required</div>
                    <div class="result-desc">The predictive model classifies this profile as: <strong>{prediction.upper()}</strong>. 
                    <br>Statistical parameters indicate a severe deviation from the success baseline. Immediate structural academic support is advised.</div>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Average":
                st.markdown(f"""
                <div class="result-block risk-avg">
                    <div class="result-title">EVALUATION: Standard Monitoring Recommended</div>
                    <div class="result-desc">The predictive model classifies this profile as: <strong>{prediction.upper()}</strong>. 
                    <br>The student aligns with standard baseline distributions. Trajectory modifications via localized academic nudging may prevent negative stabilization.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-block risk-low">
                    <div class="result-title">EVALUATION: Optimal Trajectory Confirmed</div>
                    <div class="result-desc">The predictive model classifies this profile as: <strong>{prediction.upper()}</strong>. 
                    <br>Data vectors align heavily with highest historical success rates. No structural intervention required at this juncture.</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Modify Inputs"):
                    st.session_state.step = 3
                    st.rerun()
            with col2:
                if st.button("Start New Evaluation"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
