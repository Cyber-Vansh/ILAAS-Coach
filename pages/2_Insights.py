import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ui_style

st.set_page_config(page_title="Statistical Insights", layout="wide")
ui_style.apply_german_ui()

col_main, _ = st.columns([1, 0.05])

with col_main:
    st.markdown("<h1>Statistical Insights.</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Analysis of Random Forest model parameters and historical structural data correlations.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def get_feature_importances():
        try:
            m = joblib.load("model/saved_model.pkl")
            if hasattr(m, 'feature_importances_'):
                importances = m.feature_importances_
                features = m.feature_names_in_
                df = pd.DataFrame({'Feature': features, 'Importance': importances})
                df = df.sort_values(by='Importance', ascending=False).head(10)
                return df
            return None
        except Exception:
            return None

    df_imp = get_feature_importances()
    
    if df_imp is not None:
        st.markdown("<h3>01. Feature Importance Mapping</h3>", unsafe_allow_html=True)
        st.write("The following chart delineates the top 10 variables providing the highest information gain to the predictive algorithm.")
        
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#111111'])
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family="Inter",
            font_color="#111111",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Feature importances not available for the current model architecture.")

    st.markdown("<h3>02. System Specifications</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Algorithm Setup</h4>
            <p><strong>Type:</strong> Supervised Learning / Ensemble</p>
            <p><strong>Architecture:</strong> Random Forest Classifier</p>
            <p><strong>Target Classes:</strong> Multi-class risk strata (High Risk, Medium, Low)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>Dataset Specifications</h4>
            <p><strong>Source:</strong> Portuguese Student Datasets</p>
            <p><strong>Features Analyzed:</strong> 41 (Post-One-Hot-Encoding)</p>
            <p><strong>Target Variable:</strong> G3 (Final Period Grade)</p>
        </div>
        """, unsafe_allow_html=True)
