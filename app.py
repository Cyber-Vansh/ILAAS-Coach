import streamlit as st
import ui_style
import ui_style
st.set_page_config(
    page_title="ILAAS — Intelligent Learning Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui_style.apply_german_ui()

# ---------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------
st.markdown("<h1>ILAAS</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='lead'>Intelligent Learning Analytics & Action System. "
    "An AI-powered platform that predicts student risk, diagnoses learning gaps, "
    "and generates personalized study plans using an agentic AI pipeline.</p>",
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">395</div>
        <div class="stat-label">Students Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">87%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">3</div>
        <div class="stat-label">Risk Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">AI</div>
        <div class="stat-label">Powered Study Coach</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>System Modules</h3>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="info-card">
        <h4>01. Evaluation Matrix</h4>
        <p>Input a student's academic and demographic profile to receive an instant 
        risk classification — At Risk, Average, or High Performer — powered by 
        a trained Random Forest model.</p>
        <span class="status-badge status-online">● Online</span>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="info-card">
        <h4>02. Statistical Insights</h4>
        <p>Explore the model's feature importance rankings and dataset statistics. 
        Understand which factors drive student performance the most in our ML system.</p>
        <span class="status-badge status-online">● Online</span>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="info-card">
        <h4>03. AI Study Coach ✨ New</h4>
        <p>Powered by Groq & LangGraph. Enter a student's goals and scores, and our 
        agentic AI builds a personalized learning diagnosis, 4-week study plan, 
        resource links, and a practice quiz — automatically.</p>
        <span class="status-badge status-online">● Online</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>How It Works</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

steps = [
    ("1", "Input Data", "Enter student academic grades, demographics, and environment."),
    ("2", "ML Prediction", "Random Forest model classifies the student's risk level."),
    ("3", "Agent Diagnoses", "LangGraph agent reads the result and finds learning gaps."),
    ("4", "Study Plan Ready", "A custom 4-week plan, resources, and quiz are generated."),
]

for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
    with col:
        st.markdown(f"""
        <div class="info-card" style="text-align:center;">
            <div style="font-size:2rem; font-weight:900; background: linear-gradient(135deg, #a78bfa, #38bdf8);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text; margin-bottom:0.5rem;">{num}</div>
            <h4 style="text-align:center;">{title}</h4>
            <p style="font-size:0.88rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<p style="color:#4b5563; font-size:0.82rem; text-align:center;">
    ILAAS — Milestone 2 | Built with Streamlit, scikit-learn, LangGraph & Groq API
</p>
""", unsafe_allow_html=True)
