import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the pickled Logistic Regression model
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Custom CSS for a modern black background theme
st.markdown("""
    <style>
        /* Global styles */
        body {
            background-color: #0a0a0a;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .title {
            font-size: 48px;
            color: #ffffff;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
            letter-spacing: -1px;
        }

        .subheader {
            font-size: 20px;
            color: #bbbbbb;
            font-weight: 500;
            text-align: center;
            margin-bottom: 40px;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
            text-align: center;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
        }

        .error {
            font-size: 24px;
            font-weight: bold;
            color: #e74c3c;
            text-align: center;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
        }

        .btn {
            background-color: #4a90e2;
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
            margin-top: 20px;
            width: 100%;
            max-width: 300px;
        }

        .btn:hover {
            background-color: #357ab7;
            transform: scale(1.05);
        }

        .input-field {
            margin-bottom: 20px;
            width: 100%;
        }

        .form-container {
            background-color: #1a1a1a;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 800px;
            animation: slideIn 0.8s ease-in-out;
        }

        .info-text {
            font-size: 16px;
            color: #7f8c8d;
            text-align: center;
            margin-top: 40px;
        }

        /* Input fields styling */
        .stSelectbox, .stNumberInput, .stTextInput {
            background-color: #2a2a2a;
            border: 1px solid #444444;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
            color: #ffffff;
            transition: border-color 0.3s ease;
        }

        .stSelectbox:hover, .stNumberInput:hover, .stTextInput:hover {
            border-color: #4a90e2;
        }

        /* Card shadows and borders */
        .stButton {
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.4);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<p class="title">🚀 SpaceX Falcon 9 Landing Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Predict the success of Falcon 9 first stage landings with machine learning.</p>', unsafe_allow_html=True)

st.markdown('<div class="form-container">', unsafe_allow_html=True)

st.write("### Input Features")

# Input fields for all features in a clean, structured layout
col1, col2 = st.columns(2)

with col1:
    flight_number = st.number_input('Flight Number', min_value=1, value=1, key="flight_number")
    booster_version = st.number_input('Booster Version', min_value=1, value=1, key="booster_version")
    payload_mass = st.number_input('Payload Mass (kg)', min_value=0, value=5000, key="payload_mass")
    orbit = st.selectbox('Orbit', ['LEO', 'GEO', 'MEO', 'Polar', 'SSO'], key="orbit")
    flights = st.number_input('Number of Flights', min_value=1, value=10, key="flights")
    gridfins = st.selectbox('Grid Fins', ['Yes', 'No'], key="gridfins")
    reused = st.selectbox('Reused', ['Yes', 'No'], key="reused")

with col2:
    legs = st.selectbox('Legs', ['Yes', 'No'], key="legs")
    block = st.number_input('Block', min_value=1, value=5, key="block")
    reused_count = st.number_input('Reused Count', min_value=0, value=1, key="reused_count")
    date = st.date_input('Launch Date', key="date")
    launch_site = st.selectbox('Launch Site', ['CCAFS LC-40', 'VAFB SLC-4E', 'KSC LC-39A', 'CCAFS SLC-40'], key="launch_site")
    landing_pad = st.selectbox('Landing Pad', ['LZ-1', 'LZ-2', 'OCISLY', 'ASDS'], key="landing_pad")
    longitude = st.number_input('Longitude', value=0.0, key="longitude")
    latitude = st.number_input('Latitude', value=0.0, key="latitude")

st.markdown('</div>', unsafe_allow_html=True)

# Preprocessing the input data
date_timestamp = pd.to_datetime(date)
date_numeric = (date_timestamp - pd.to_datetime('1970-01-01')).days

# Categorical to numeric mapping
orbit_mapping = {'LEO': 0, 'GEO': 1, 'MEO': 2, 'Polar': 3, 'SSO': 4}
launch_site_mapping = {'CCAFS LC-40': 0, 'VAFB SLC-4E': 1, 'KSC LC-39A': 2, 'CCAFS SLC-40': 3}
gridfins_mapping = {'Yes': 1, 'No': 0}
reused_mapping = {'Yes': 1, 'No': 0}
legs_mapping = {'Yes': 1, 'No': 0}
landing_pad_mapping = {'LZ-1': 0, 'LZ-2': 1, 'OCISLY': 2, 'ASDS': 3}

# Prepare input data for prediction
input_data = np.array([[ 
    flight_number,
    date_numeric,
    booster_version,
    payload_mass,
    orbit_mapping[orbit],
    launch_site_mapping[launch_site],
    flights,
    gridfins_mapping[gridfins],
    reused_mapping[reused],
    legs_mapping[legs],
    landing_pad_mapping[landing_pad],
    block,
    reused_count,
    longitude,
    latitude
]])

# Make prediction using the loaded model
if st.button('Predict Landing Outcome', key='predict_btn', use_container_width=True):
    try:
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)

        result = 'Successful' if prediction[0] == 1 else 'Unsuccessful'
        prob = '{:.2f}%'.format(max(prediction_prob[0]) * 100)
        
        st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction">Confidence: {prob}</p>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<p class="error">Error during prediction: {str(e)}</p>', unsafe_allow_html=True)