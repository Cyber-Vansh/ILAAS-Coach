#!/usr/bin/env bash

set -e

# Navigate to the script's directory (project root)
cd "$(dirname "$0")"

# Create and activate a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

