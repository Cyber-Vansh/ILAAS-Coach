import streamlit as st
import ui_style

st.set_page_config(page_title="ILAAS SYSTEM.", page_icon="🏛", layout="wide")
ui_style.apply_german_ui()

col_main, _ = st.columns([1, 0.05])

with col_main:
    st.markdown("<h1>System Overview.</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Intelligent Learning Analytics & Action System (ILAAS). Developed for institutional academic diagnostics.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3>System Modules</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>01. Evaluation Matrix</h4>
            <p>Access the predictive neural engine. Input demographic and academic vectors to receive rapid, data-driven trajectory forecasts for individual students.</p>
            <p><strong>Status:</strong> Online</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>02. Statistical Insights</h4>
            <p>Review the macro-level correlations and historical dataset parameters that drive the predictive engine's internal weights.</p>
            <p><strong>Status:</strong> Online</p>
        </div>
        """, unsafe_allow_html=True)
