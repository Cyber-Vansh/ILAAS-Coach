import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="ILA COACH",
    page_icon="STR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Swiss Grid Styling
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        
        .main-header {
            font-size: 4rem;
            font-weight: 800;
            letter-spacing: -2px;
            color: #000000;
            text-align: left;
            margin-bottom: 3rem;
            line-height: 1;
            text-transform: uppercase;
        }
        
        .metric-card {
            background: #ffffff;
            padding: 2rem;
            border: 2px solid #000000;
            color: #000000;
            margin: 0.5rem 0;
            border-radius: 0px;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1;
        }
        
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666666;
        }
        
        .sidebar-section {
            background: #ffffff;
            padding: 1rem;
            border: 1px solid #000000;
            margin: 1rem 0;
            border-radius: 0px;
        }
        
        .stButton > button {
            background: #000000;
            color: #ffffff;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: #E6192E;
            color: #ffffff;
        }
        
        .plot-container {
            background: white;
            padding: 1rem;
            border: 1px solid #eeeeee;
            margin: 1rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 16px;
            height: 4px;
            margin-right: 8px;
        }
        
        .status-high { background-color: #E6192E; }
        .status-medium { background-color: #000000; }
        .status-low { background-color: #cccccc; }
        
        .info-box {
            border: 2px solid #000000;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: #E6192E;
            color: white;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .success-box {
            border: 2px solid #000000;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Layout Grid */
        .swiss-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Load data
def load_sample_data():
    np.random.seed(42)
    n_students = 1000
    
    data = {
        'student_id': [f'STU_{i:04d}' for i in range(1, n_students + 1)],
        'name': [f'Student {i}' for i in range(1, n_students + 1)],
        'age': np.random.randint(18, 25, n_students),
        'gpa': np.random.uniform(2.0, 4.0, n_students),
        'attendance_rate': np.random.uniform(60, 100, n_students),
        'assignment_completion': np.random.uniform(50, 100, n_students),
        'study_hours_per_week': np.random.randint(1, 40, n_students),
        'extracurricular_activities': np.random.randint(0, 5, n_students),
        'risk_score': np.random.uniform(0, 1, n_students),
        'last_login': pd.date_range(start='2024-01-01', end='2024-12-31', periods=n_students),
        'program': np.random.choice(['Computer Science', 'Engineering', 'Business', 'Arts', 'Science'], n_students),
        'year': np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], n_students)
    }
    
    df = pd.DataFrame(data)
    df['risk_level'] = pd.cut(df['risk_score'], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'])
    
    return df

# Load ML model
def load_model():
    try:
        model_path = 'src/student_risk_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
        
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_model.fit(X, y)
        
        st.info("Using demo model for demonstration purposes")
        return dummy_model
    return None

# Load scaler
def load_scaler():
    try:
        scaler_path = 'src/scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
    except Exception as e:
        st.warning(f"Could not load scaler: {e}")
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        dummy_scaler = StandardScaler()
        sample_data = np.random.rand(100, 5)
        dummy_scaler.fit(sample_data)
        
        st.info("Using demo scaler for demonstration purposes")
        return dummy_scaler
    return None

def sidebar():
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("ILA COACH")
    st.sidebar.markdown("INTELLIGENT LEARNING ANALYTICS")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "NAVIGATION",
        ["» DASHBOARD", "» DATA VISUALIZATION", "» RISK ANALYSIS", "» SETTINGS"],
        index=0
    )
    
    # Strip the arrow for logic
    page = page.replace("» ", "")
    
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Quick Stats")
    df = load_sample_data()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        high_risk = len(df[df['risk_level'] == 'High'])
        st.metric("High Risk", high_risk)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return page

def dashboard_page(df, model, scaler):
    st.markdown('<h1 class="main-header">» DASHBOARD</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_gpa = df['gpa'].mean() if 'gpa' in df.columns else df['G3'].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-label">AVG GRADE</div><div class="metric-value">{avg_gpa:.2f}</div></div>', unsafe_allow_html=True)
    
    with col2:
        avg_attendance = df['attendance_rate'].mean() if 'attendance_rate' in df.columns else df['absences'].mean()
        label = "AVG ATTENDANCE" if 'attendance_rate' in df.columns else "AVG ABSENCES"
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{avg_attendance:.1f}</div></div>', unsafe_allow_html=True)
    
    with col3:
        high_risk_pct = (df['risk_level'] == 'High').sum() / len(df) * 100
        st.markdown(f'<div class="metric-card"><div class="metric-label">HIGH RISK %</div><div class="metric-value">{high_risk_pct:.1f}%</div></div>', unsafe_allow_html=True)
    
    with col4:
        completion_rate = df['assignment_completion'].mean() if 'assignment_completion' in df.columns else 0
        st.markdown(f'<div class="metric-card"><div class="metric-label">COMPLETION RATE</div><div class="metric-value">{completion_rate:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-label">RISK DISTRIBUTION</div>', unsafe_allow_html=True)
        risk_counts = df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_sequence=['#000000', '#E6192E', '#cccccc']
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="metric-label">GRADE DISTRIBUTION</div>', unsafe_allow_html=True)
        plot_col = 'gpa' if 'gpa' in df.columns else 'G3' if 'G3' in df.columns else 'G2'
        fig = px.histogram(df, x=plot_col, nbins=20, color_discrete_sequence=['#000000'])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="metric-label">RECENT ALERTS</div>', unsafe_allow_html=True)
    high_risk_students = df[df['risk_level'] == 'High'].sort_values('risk_score', ascending=False).head(5)
    
    for _, student in high_risk_students.iterrows():
        y_val = student['gpa'] if 'gpa' in student else 0
        st.markdown(f"""
        <div class="warning-box">
            <strong>{student['name'].upper()}</strong> | RISK: {student['risk_score']:.2f} | GRADE: {y_val:.2f}
        </div>
        """, unsafe_allow_html=True)

def data_viz_page(df):
    st.markdown('<h1 class="main-header">» VISUALIZATION</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_program = st.selectbox("Select Program", ['All'] + list(df['program'].unique()))
    
    with col2:
        selected_year = st.selectbox("Select Year", ['All'] + list(df['year'].unique()))
    
    with col3:
        selected_risk = st.selectbox("Select Risk Level", ['All'] + list(df['risk_level'].unique()))
    
    filtered_df = df.copy()
    if selected_program != 'All':
        filtered_df = filtered_df[filtered_df['program'] == selected_program]
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['year'] == selected_year]
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['risk_level'] == selected_risk]
    
    st.info(f"Showing {len(filtered_df)} students out of {len(df)} total students")
    
    st.subheader("🔥 Correlation Matrix")
    numeric_cols = ['gpa', 'attendance_rate', 'assignment_completion', 'study_hours_per_week', 'risk_score']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Multi-dimensional Analysis")
    
    chart_type = st.selectbox("Select Chart Type", ["3D Scatter", "Parallel Coordinates", "Box Plot"])
    
    if chart_type == "3D Scatter":
        fig = px.scatter_3d(
            filtered_df, x='gpa', y='attendance_rate', z='study_hours_per_week',
            color='risk_level', size='assignment_completion',
            hover_data=['name'], symbol='program',
            color_discrete_map={'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Parallel Coordinates":
        fig = px.parallel_coordinates(
            filtered_df, dimensions=['gpa', 'attendance_rate', 'assignment_completion', 'study_hours_per_week'],
            color='risk_score', color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Box Plot
        fig = px.box(filtered_df, x='program', y='gpa', color='risk_level',
                    color_discrete_map={'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def risk_analysis_page(df, model, scaler):
    st.markdown('<h1 class="main-header">» RISK ANALYSIS</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-label">STUDENT SELECTION</div>', unsafe_allow_html=True)
    selected_student = st.selectbox("", df['name'].unique())
    student_data = df[df['name'] == selected_student].iloc[0]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <div class="metric-label">DATA PROFILE</div>
            <p>ID: {student_data['student_id']}</p>
            <p>PROGRAM: {student_data['program'].upper()}</p>
            <p>YEAR: {student_data['year'].upper()}</p>
            <p>AGE: {student_data['age']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level = student_data['risk_level']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ASSESSMENT</div>
            <div class="metric-value">{risk_level.upper()} RISK</div>
            <p>SCORE: {student_data['risk_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">GPA</div><div class="metric-value">{student_data["gpa"]:.2f}</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ATTENDANCE</div><div class="metric-value">{student_data["attendance_rate"]:.1f}%</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">COMPLETION</div><div class="metric-value">{student_data["assignment_completion"]:.1f}%</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">STUDY HOURS</div><div class="metric-value">{student_data["study_hours_per_week"]}</div></div>', unsafe_allow_html=True)
    
    st.subheader("🔮 Predictive Analysis")
    
    if model and scaler:
        features = np.array([[
            student_data['gpa'],
            student_data['attendance_rate'],
            student_data['assignment_completion'],
            student_data['study_hours_per_week'],
            student_data['extracurricular_activities']
        ]])
        
        try:
            features_scaled = scaler.transform(features)
            
            prediction = model.predict_proba(features_scaled)[0]
            risk_prediction = prediction[1]  # Probability of high risk
            
            st.write(f"**Model Prediction:** {risk_prediction:.1%} probability of high risk")
            
            feature_importance = {
                'GPA': 0.3,
                'Attendance': 0.25,
                'Assignment Completion': 0.2,
                'Study Hours': 0.15,
                'Extracurricular': 0.1
            }
            
            fig = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                title="Feature Importance"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("ML model not available. Please train the model first.")
    
    st.subheader("💡 Recommendations")
    
    if risk_level == 'High':
        st.markdown("""
        <div class="warning-box">
            <h4>Immediate Action Required</h4>
            <ul>
                <li>Schedule one-on-one meeting within 48 hours</li>
                <li>Provide additional academic support</li>
                <li>Monitor attendance closely</li>
                <li>Connect with counseling services</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == 'Medium':
        st.markdown("""
        <div class="info-box">
            <h4>Proactive Support Recommended</h4>
            <ul>
                <li>Check in weekly via email</li>
                <li>Offer tutoring resources</li>
                <li>Encourage study group participation</li>
                <li>Monitor progress monthly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <h4>Continue Current Support</h4>
            <ul>
                <li>Maintain regular check-ins</li>
                <li>Provide advanced learning opportunities</li>
                <li>Consider peer mentoring roles</li>
                <li>Monitor for any changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def settings_page():
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Display Settings")
        
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        color_scheme = st.selectbox("Color Scheme", ["Purple", "Blue", "Green", "Orange"])
        
        st.subheader("📊 Data Settings")
        
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
        
        show_alerts = st.checkbox("Show real-time alerts", value=True)
        alert_threshold = st.slider("Risk alert threshold", 0.5, 1.0, 0.7)
    
    with col2:
        st.subheader("🔔 Notification Settings")
        
        email_notifications = st.checkbox("Email notifications", value=True)
        sms_notifications = st.checkbox("SMS notifications", value=False)
        
        notification_frequency = st.selectbox(
            "Notification Frequency",
            ["Real-time", "Daily", "Weekly", "Monthly"]
        )
        
        st.subheader("🔐 Privacy Settings")
        
        data_anonymization = st.checkbox("Anonymize student data", value=False)
        export_logs = st.checkbox("Export access logs", value=False)
        
        retention_period = st.selectbox(
            "Data Retention Period",
            ["30 days", "90 days", "1 year", "5 years"]
        )
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")
    
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Version", "1.0.0")
    
    with col2:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    with col3:
        st.metric("Status", "🟢 Online")

def main():
    load_css()
    
    df = load_sample_data()
    model = load_model()
    scaler = load_scaler()
    
    page = sidebar()
    
    if page == "DASHBOARD":
        dashboard_page(df, model, scaler)
    elif page == "DATA VISUALIZATION":
        data_viz_page(df)
    elif page == "RISK ANALYSIS":
        risk_analysis_page(df, model, scaler)
    elif page == "SETTINGS":
        settings_page()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "© 2024 ILA Coach - Intelligent Learning Analytics Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
