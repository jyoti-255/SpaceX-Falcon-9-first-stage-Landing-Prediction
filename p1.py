import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv('SpaceX_Falcon9.csv')

# Handle missing values
print(data.isnull().sum())

data['PayloadMass'].fillna(data['PayloadMass'].mean(), inplace=True)
data['LandingPad'].fillna('Unknown', inplace=True)

print(data.isnull().sum())

# Feature Engineering
categorical_columns = ["Orbit", "LaunchSite", "LandingPad", "BoosterVersion", "Serial"]
features = data[[
    "FlightNumber", "PayloadMass", "Orbit", "LaunchSite", "Flights", 
    "GridFins", "Reused", "Legs", "LandingPad", "Block", 
    "ReusedCount", "Longitude", "Latitude", "BoosterVersion", "Serial"
]]

# One-hot encoding for categorical variables
features_one_hot = pd.get_dummies(features, columns=categorical_columns, drop_first=True)
print(features_one_hot)

#check is there any null value present now
print("\nMissing Values After One-Hot Encoding:")
print(features_one_hot.isnull().sum().sum())

# Define features (X) and target (Y)
X = features_one_hot
Y = data["Outcome"].apply(lambda x: 1 if "True" in str(x) else 0).to_numpy()

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
