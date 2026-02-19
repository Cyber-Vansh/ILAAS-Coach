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
    page_title="ILA Coach Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            color: white;
            margin: 0.5rem 0;
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .plot-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-high { background-color: #ff4757; }
        .status-medium { background-color: #ffa502; }
        .status-low { background-color: #26de81; }
        
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        .warning-box {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        .success-box {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
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
    st.sidebar.title("🎓 ILA Coach")
    st.sidebar.markdown("Intelligent Learning Analytics Coach")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "📈 Data Visualization", "🎯 Risk Analysis", "⚙️ Settings"],
        index=0
    )
    
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
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_gpa = df['gpa'].mean()
        st.metric("Average GPA", f"{avg_gpa:.2f}", delta="↑ 0.1")
    
    with col2:
        avg_attendance = df['attendance_rate'].mean()
        st.metric("Avg Attendance", f"{avg_attendance:.1f}%", delta="↑ 2.3%")
    
    with col3:
        high_risk_pct = (df['risk_level'] == 'High').sum() / len(df) * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%", delta="↓ 1.2%")
    
    with col4:
        completion_rate = df['assignment_completion'].mean()
        st.metric("Completion Rate", f"{completion_rate:.1f}%", delta="↑ 3.5%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 GPA Distribution")
        fig = px.histogram(df, x='gpa', nbins=20, color='risk_level',
                          color_discrete_map={'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Attendance vs GPA")
        fig = px.scatter(df, x='attendance_rate', y='gpa', color='risk_level',
                        size='study_hours_per_week', hover_data=['name'],
                        color_discrete_map={'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎓 Program Distribution")
        program_counts = df['program'].value_counts()
        fig = px.bar(x=program_counts.index, y=program_counts.values,
                    labels={'x': 'Program', 'y': 'Number of Students'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🚨 Recent Alerts")
    high_risk_students = df[df['risk_level'] == 'High'].sort_values('risk_score', ascending=False).head(5)
    
    for _, student in high_risk_students.iterrows():
        risk_color = '#ff4757'
        st.markdown(f"""
        <div class="warning-box">
            <span class="status-indicator status-high"></span>
            <strong>{student['name']}</strong> - Risk Score: {student['risk_score']:.2f}<br>
            GPA: {student['gpa']:.2f} | Attendance: {student['attendance_rate']:.1f}% | 
            Program: {student['program']}
        </div>
        """, unsafe_allow_html=True)

def data_viz_page(df):
    st.markdown('<h1 class="main-header">📈 Data Visualization</h1>', unsafe_allow_html=True)
    
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
    st.markdown('<h1 class="main-header">🎯 Risk Analysis</h1>', unsafe_allow_html=True)
    
    st.subheader("👤 Individual Student Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_student = st.selectbox("Select Student", df['name'].unique())
        student_data = df[df['name'] == selected_student].iloc[0]
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Student Information</h4>
            <p><strong>ID:</strong> {student_data['student_id']}</p>
            <p><strong>Program:</strong> {student_data['program']}</p>
            <p><strong>Year:</strong> {student_data['year']}</p>
            <p><strong>Age:</strong> {student_data['age']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Risk level indicator
        risk_level = student_data['risk_level']
        risk_color = {'Low': '#26de81', 'Medium': '#ffa502', 'High': '#ff4757'}[risk_level]
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Assessment</h3>
            <h2 style="color: {risk_color};">{risk_level} Risk</h2>
            <p>Risk Score: {student_data['risk_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("GPA", f"{student_data['gpa']:.2f}")
    
    with col2:
        st.metric("Attendance", f"{student_data['attendance_rate']:.1f}%")
    
    with col3:
        st.metric("Assignment Completion", f"{student_data['assignment_completion']:.1f}%")
    
    with col4:
        st.metric("Study Hours/Week", student_data['study_hours_per_week'])
    
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
    
    if page == "📊 Dashboard":
        dashboard_page(df, model, scaler)
    elif page == "📈 Data Visualization":
        data_viz_page(df)
    elif page == "🎯 Risk Analysis":
        risk_analysis_page(df, model, scaler)
    elif page == "⚙️ Settings":
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
