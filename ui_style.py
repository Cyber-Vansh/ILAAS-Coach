import streamlit as st

# This function applies our custom CSS styles to every page
def apply_german_ui():
    st.markdown("""
    <style>
        /* ============================================================
           Import Google Font: Inter (clean, modern, professional)
        ============================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ============================================================
           Hide Streamlit default UI elements we don't need
        ============================================================ */
        #MainMenu, footer, header { visibility: hidden; display: none; }

        /* Keep the sidebar toggle button visible */
        [data-testid="collapsedControl"] {
            display: flex !important;
            z-index: 1000000 !important;
        }

        /* ============================================================
           Global base styles — dark background, white text
        ============================================================ */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            font-family: 'Inter', sans-serif;
            background-color: #0a0a0f;
            color: #e8e8f0;
        }

        /* The main content block */
        [data-testid="stAppViewContainer"] > .main {
            background-color: #0a0a0f;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d18 0%, #0a0a14 100%);
            border-right: 1px solid rgba(139, 92, 246, 0.2);
        }

        /* ============================================================
           Typography — Headings
        ============================================================ */
        h1 {
            font-family: 'Inter', sans-serif;
            font-weight: 900;
            font-size: 3rem;
            line-height: 1.1;
            letter-spacing: -0.03em;
            /* Purple to cyan gradient text */
            background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            color: #e8e8f0;
            letter-spacing: -0.02em;
        }

        h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            color: #e8e8f0;
            margin-top: 2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        h4 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #a78bfa;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.8rem;
        }

        p {
            color: #9ca3af;
            line-height: 1.6;
        }

        /* Lead paragraph (subtitle under h1) */
        p.lead {
            font-size: 1.1rem;
            color: #9ca3af;
            font-weight: 400;
            margin-bottom: 2.5rem;
            max-width: 650px;
            line-height: 1.7;
        }

        /* ============================================================
           Cards — glassmorphism style with subtle border
        ============================================================ */
        .info-card {
            background: rgba(139, 92, 246, 0.05);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.25rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .info-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #a78bfa, #38bdf8);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .info-card:hover {
            border-color: rgba(139, 92, 246, 0.5);
            background: rgba(139, 92, 246, 0.08);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(139, 92, 246, 0.15);
        }

        .info-card:hover::before {
            opacity: 1;
        }

        .info-card h4 {
            color: #a78bfa;
        }

        .info-card p {
            color: #9ca3af;
            margin: 0;
            font-size: 0.95rem;
        }

        /* Status badge inside cards */
        .status-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.75rem;
        }

        .status-online {
            background: rgba(52, 211, 153, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        /* ============================================================
           Stat cards — for the homepage numbers
        ============================================================ */
        .stat-card {
            background: rgba(15, 15, 30, 0.8);
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #a78bfa, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }

        .stat-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.5rem;
        }

        /* ============================================================
           Result blocks — for ML predictions (colored left border)
        ============================================================ */
        .result-block {
            border-left: 4px solid #a78bfa;
            border-radius: 0 12px 12px 0;
            padding: 2rem;
            margin-top: 2rem;
            background: rgba(15, 15, 30, 0.8);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.4s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-weight: 800;
            font-size: 1.4rem;
            color: #e8e8f0;
            margin-bottom: 0.75rem;
            letter-spacing: -0.01em;
        }

        .result-desc {
            font-size: 1rem;
            color: #9ca3af;
            line-height: 1.7;
        }

        /* Risk color variants */
        .risk-high { border-left-color: #f87171; }
        .risk-avg  { border-left-color: #fbbf24; }
        .risk-low  { border-left-color: #34d399; }

        /* ============================================================
           Wizard step indicator
        ============================================================ */
        .wizard-step {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #a78bfa;
            margin-bottom: 0.75rem;
        }

        /* ============================================================
           Streamlit widget overrides — inputs, sliders, buttons
        ============================================================ */
        /* All input boxes */
        div.stSelectbox > div > div > div,
        div.stNumberInput > div > div > div,
        div.stTextInput > div > div > input,
        div.stTextArea > div > div > textarea {
            background-color: rgba(15, 15, 30, 0.8) !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            border-radius: 10px !important;
            color: #e8e8f0 !important;
            font-family: 'Inter', sans-serif !important;
            transition: border-color 0.2s ease !important;
        }

        div.stSelectbox > div > div > div:focus-within,
        div.stTextInput > div > div > input:focus,
        div.stTextArea > div > div > textarea:focus {
            border-color: #a78bfa !important;
            box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.15) !important;
        }

        /* Primary button (purple gradient) */
        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.5rem !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }

        .stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
        }

        /* Secondary button (outline) */
        .stButton > button {
            background: rgba(139, 92, 246, 0.08) !important;
            color: #a78bfa !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            border-radius: 10px !important;
            padding: 0.65rem 1.5rem !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.2s ease !important;
            width: 100%;
        }

        .stButton > button:hover {
            background: rgba(139, 92, 246, 0.15) !important;
            border-color: #a78bfa !important;
        }

        /* Slider track color */
        div.stSlider > div > div > div > div {
            background: linear-gradient(90deg, #7c3aed, #38bdf8) !important;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #7c3aed, #38bdf8) !important;
        }

        /* ============================================================
           Agent output sections
        ============================================================ */
        .agent-section {
            background: rgba(15, 15, 30, 0.8);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            animation: fadeUp 0.5s ease;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(15px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .agent-section-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #a78bfa;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .agent-section-body {
            color: #d1d5db;
            font-size: 0.95rem;
            line-height: 1.8;
            white-space: pre-wrap;
        }

        /* ============================================================
           Nav pill (sidebar page header)
        ============================================================ */
        .nav-brand {
            font-size: 1.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a78bfa, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }

        /* Plotly chart dark background fix */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }

        /* Spinner text color */
        .stSpinner > div {
            color: #a78bfa !important;
        }

        /* Divider */
        hr {
            border-color: rgba(139, 92, 246, 0.2) !important;
            margin: 2rem 0 !important;
        }

        /* Info/warning boxes */
        .stAlert {
            border-radius: 10px !important;
            border-left-color: #a78bfa !important;
        }
    </style>
    """, unsafe_allow_html=True)
