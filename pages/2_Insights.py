import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ui_style

import ui_style

st.set_page_config(page_title="Statistical Insights — ILAAS", page_icon="📉", layout="wide")
ui_style.apply_german_ui()

st.markdown("<h1>Statistical Insights</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='lead'>Explore the Random Forest model's feature importances and dataset statistics. "
    "Understand exactly which factors drive student success.</p>",
    unsafe_allow_html=True
)

)

@st.cache_resource
def get_feature_importances():
    try:
        m = joblib.load("api/model/saved_model.pkl")
        if hasattr(m, 'feature_importances_'):
            importances = m.feature_importances_
            features = m.feature_names_in_
            df = pd.DataFrame({'Feature': features, 'Importance': importances})
            df = df.sort_values(by='Importance', ascending=False).head(15)
            return df, m
        return None, m
    except Exception:
        return None, None

df_imp, model = get_feature_importances()

df_imp, model = get_feature_importances()

if df_imp is not None:
    st.markdown("<h3>📊 Feature Importance Ranking</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>The chart below shows which input variables have the highest impact "
        "on the model's risk prediction. A higher bar means the model relies more "
        "heavily on that feature when making decisions.</p>",
        unsafe_allow_html=True
    )

    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[[0, '#4c1d95'], [0.5, '#7c3aed'], [1.0, '#38bdf8']],
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
    )

    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 20, 0.0)',
        paper_bgcolor='rgba(10, 10, 20, 0.0)',
        font_family="Inter",
        font_color="#9ca3af",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        xaxis=dict(
            gridcolor='rgba(139, 92, 246, 0.1)',
            tickfont=dict(color='#6b7280')
        ),
        yaxis_title=None,
        xaxis_title='Importance Score',
    )
    fig.update_traces(marker_line_color='rgba(0,0,0,0)', marker_line_width=0)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Feature importances not available. Please make sure the model is trained.")

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>🤖 Model Specifications</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>Algorithm</h4>
        <p><strong>Type:</strong> Ensemble / Supervised Learning</p>
        <p><strong>Architecture:</strong> Random Forest Classifier</p>
        <p><strong>Risk Classes:</strong> At-Risk · Average · High Performer</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>Dataset</h4>
        <p><strong>Source:</strong> UCI Student Performance</p>
        <p><strong>Language:</strong> Portuguese Student Dataset</p>
        <p><strong>Features Used:</strong> 41 (post one-hot encoding)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h4>Performance</h4>
        <p><strong>Accuracy:</strong> ~87%</p>
        <p><strong>Risk Threshold:</strong> ≥15% probability for At-Risk</p>
        <p><strong>Method:</strong> predict_proba() thresholding</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3>💡 Feature Explanations</h3>", unsafe_allow_html=True)
st.markdown(
    "<p>Below is a plain-English explanation of the most important features the model uses.</p>",
    unsafe_allow_html=True
)

feature_descriptions = {
    "G2": "Semester 2 grade — the strongest predictor of final performance.",
    "G1": "Semester 1 grade — combined with G2, gives a strong trajectory signal.",
    "failures": "Number of past class failures — directly correlated with higher risk.",
    "absences": "Number of school absences — high absences strongly predict poor outcomes.",
    "studytime": "Weekly study hours at home — more study time correlates with success.",
    "age": "Student age — older students sometimes correlate with more failures.",
    "higher_yes": "Plans to attend higher education — a strong motivational indicator.",
    "internet_yes": "Has internet access at home — correlated with academic resources.",
    "schoolsup_yes": "Receives extra school support — may indicate prior struggle.",
    "Medu": "Mother's education level — household education strongly impacts students.",
}

table_rows = []
for feature, description in feature_descriptions.items():
    table_rows.append({"Feature": feature, "Plain-English Meaning": description})

df_desc = pd.DataFrame(table_rows)

st.dataframe(
    df_desc,
    use_container_width=True,
    hide_index=True,
)
