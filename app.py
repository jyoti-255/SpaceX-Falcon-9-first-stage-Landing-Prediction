import pandas as pd
import numpy as np
import pickle
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")



# Load the trained model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit App
st.set_page_config(page_title="SpaceX Falcon 9 Prediction", page_icon="🚀", layout="wide")
st.title("🚀 SpaceX Falcon 9 Landing Prediction")
st.sidebar.markdown("## Upload a CSV File")

# File uploader in sidebar
dataset = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if dataset is not None:
    data = pd.read_csv(dataset)
    
    # Convert Date column to datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    st.write("### 📊 Exploratory Data Analysis (EDA)")

    # Display basic dataset details
    st.write("#### 📜 Dataset Overview")
    st.dataframe(data.describe())
    
    # Distribution of Payload Mass
    st.write("#### 📈 Distribution of Payload Mass")
    fig, ax = plt.subplots()
    sns.histplot(data["PayloadMass"], kde=True, bins=30, color='blue', ax=ax)
    st.pyplot(fig)
    
    # PayloadMass vs. FlightNumber
    st.write("#### 🚀 Payload Mass vs. Flight Number")
    fig, ax = plt.subplots()
    sns.scatterplot(y="PayloadMass", x="FlightNumber", hue="Outcome", data=data, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # FlightNumber vs. Launch Site
    st.write("#### 🛰️ Flight Number vs. Launch Site")
    fig, ax = plt.subplots()
    sns.boxplot(x="LaunchSite", y="FlightNumber", data=data, palette="Set2", ax=ax)
    st.pyplot(fig)

    # PayloadMass vs. Launch Site
    st.write("#### 🔥 Payload Mass vs. Launch Site")
    fig, ax = plt.subplots()
    sns.violinplot(x="LaunchSite", y="PayloadMass", data=data, palette="cool", ax=ax)
    st.pyplot(fig)

    # Success rate by Orbit Type
    success_mapping = {
        'True Ocean': 1, 'True ASDS': 1, 'True RTLS': 1,
        'False Ocean': 0, 'False ASDS': 0, 'False RTLS': 0,
        'None ASDS': 0, 'None None': 0
    }
    data['Outcome'] = data['Outcome'].map(success_mapping).fillna(0).astype(int)

    st.write("#### 🌎 Success Rate by Orbit Type")
    orbit_success_rate = data.groupby('Orbit')['Outcome'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="Orbit", y="Outcome", data=orbit_success_rate, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Correlation Heatmap (Exclude non-numeric columns)
    st.write("#### 📌 Feature Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Yearly Success Rate
    st.write("#### 📅 Success Rate Over the Years")
    yearly_success_rate = data.groupby(data['Date'].dt.year)['Outcome'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(yearly_success_rate['Date'], yearly_success_rate['Outcome'], marker='o', color='red')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("## 🎯 Prediction Section")
    st.markdown("Fill the details below to predict the Falcon 9 landing outcome:")

    # User input fields for prediction
    flight_number = st.number_input("Flight Number", min_value=1, step=1)
    payload_mass = st.number_input("Payload Mass (kg)", min_value=0.0, step=0.1)
    flights = st.number_input("Previous Flights", min_value=0, step=1)
    grid_fins = st.selectbox("Grid Fins", [0, 1])
    reused = st.selectbox("Reused Booster", [0, 1])
    legs = st.selectbox("Landing Legs", [0, 1])
    block = st.number_input("Block Version", min_value=0, step=1)
    reused_count = st.number_input("Reused Count", min_value=0, step=1)
    longitude = st.number_input("Launch Longitude", min_value=-180.0, max_value=180.0, step=0.1)
    latitude = st.number_input("Launch Latitude", min_value=-90.0, max_value=90.0, step=0.1)

    # Categorical inputs
    orbit = st.selectbox("Orbit Type", ["GTO", "LEO", "SSO", "VLEO", "ISS"])
    launch_site = st.selectbox("Launch Site", ["CCAFS", "KSC", "VAFB"])
    landing_pad = st.selectbox("Landing Pad", ["OCISLY", "JRTI", "LZ-1", "Unknown"])
    booster_version = st.selectbox("Booster Version", ["B1049", "B1051", "B1056", "B1060"])
    serial = st.selectbox("Booster Serial", ["B0003", "B0005", "B0007", "B0015"])

    # Convert categorical variables to one-hot encoding
    input_data = pd.DataFrame({
        "FlightNumber": [flight_number],
        "PayloadMass": [payload_mass],
        "Flights": [flights],
        "GridFins": [grid_fins],
        "Reused": [reused],
        "Legs": [legs],
        "Block": [block],
        "ReusedCount": [reused_count],
        "Longitude": [longitude],
        "Latitude": [latitude],
        "Orbit_" + orbit: [1],
        "LaunchSite_" + launch_site: [1],
        "LandingPad_" + landing_pad: [1],
        "BoosterVersion_" + booster_version: [1],
        "Serial_" + serial: [1]
    })

    # Ensure all expected columns exist
    expected_columns = scaler.get_feature_names_out()
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with 0

    # Reorder columns to match training data
    input_data = input_data[expected_columns]

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Predict
    if st.button("🚀 Predict Landing Outcome"): 
        prediction = model.predict(scaled_input)
        outcome = "✅ Landed Successfully" if prediction[0] == 1 else "❌ Did Not Land"
        st.success(f"Prediction: {outcome}")
