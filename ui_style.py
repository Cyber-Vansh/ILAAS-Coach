import streamlit as st

def apply_german_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap');
        
        html, body {
            font-family: 'Inter', Helvetica, sans-serif;
            letter-spacing: -0.02em;
        }
        
        #MainMenu, footer {visibility: hidden; display: none;}
        
        [data-testid="collapsedControl"] {
            display: flex !important;
            z-index: 1000000 !important;
        }
        
        h1 {
            font-weight: 900;
            font-size: 3.5rem;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            color: var(--text-color);
        }
        
        p.lead {
            font-size: 1.25rem;
            opacity: 0.8;
            font-weight: 400;
            margin-bottom: 3rem;
            max-width: 600px;
        }
        
        h3 {
            font-weight: 700;
            font-size: 1.5rem;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--text-color);
            color: var(--text-color);
        }
        
        div.stSelectbox > div > div > div, 
        div.stNumberInput > div > div > div,
        input {
            border-radius: 0px !important;
            border: 1px solid var(--text-color) !important;
            font-family: inherit !important;
            box-shadow: none !important;
            transition: border 0.2s ease;
            opacity: 0.9;
        }
        
        .stButton>button {
            border-radius: 0px;
            padding: 1rem 2rem;
            font-size: 1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 2px solid var(--text-color);
            width: 100%;
            margin-top: 1rem;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            opacity: 0.8;
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
        }
        
        .result-block {
            border-left: 6px solid var(--text-color);
            padding: 2rem;
            margin-top: 3rem;
            background-color: var(--secondary-background-color);
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        .result-title {
            font-weight: 900;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .result-desc {
            font-size: 1.1rem;
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .risk-high { border-left-color: #E60000; }
        .risk-avg { border-left-color: #FF9900; }
        .risk-low { border-left-color: #00B300; }
        
        .info-card {
            border: 1px solid var(--text-color);
            padding: 2rem;
            margin-bottom: 2rem;
            background-color: var(--secondary-background-color);
            opacity: 0.95;
        }
        .info-card h4 {
            font-weight: 900;
            font-size: 1.25rem;
            margin-top: 0;
            margin-bottom: 1rem;
        }
        
        .wizard-step {
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)
