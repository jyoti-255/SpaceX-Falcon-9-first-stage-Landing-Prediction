# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Step 2: Load the dataset
data = pd.read_csv('SpaceX_Falcon9.csv')

# Display the first few rows of the dataset
print(data.head())

# Step 3: Handle missing values
print(data.isnull().sum())
data['PayloadMass'] = data['PayloadMass'].fillna(data['PayloadMass'].median())
data['Outcome'] = data['Outcome'].fillna(data['Outcome'].mode()[0])
data['BoosterVersion'] = data['BoosterVersion'].fillna(data['BoosterVersion'].mode()[0])
data['LaunchSite'] = data['LaunchSite'].fillna(data['LaunchSite'].mode()[0])
data['Orbit'] = data['Orbit'].fillna(data['Orbit'].mode()[0])
data['LandingPad'] = data['LandingPad'].fillna(data['LandingPad'].mode()[0])
data['Block'] = data['Block'].fillna(data['Block'].mode()[0])
data['Serial'] = data['Serial'].fillna(data['Serial'].mode()[0])

# Check if missing values are handled
print(data.isnull().sum())

# Step 4: Encode categorical variables using LabelEncoder
le = LabelEncoder()

data['Outcome'] = le.fit_transform(data['Outcome'])
data['BoosterVersion'] = le.fit_transform(data['BoosterVersion'])
data['LaunchSite'] = le.fit_transform(data['LaunchSite'])
data['Orbit'] = le.fit_transform(data['Orbit'])
data['LandingPad'] = le.fit_transform(data['LandingPad'])
data['Block'] = le.fit_transform(data['Block'])
data['Serial'] = le.fit_transform(data['Serial'])

# Display first few rows after encoding
print(data.head())

print(data['LaunchSite'].value_counts())

print(data["Orbit"].value_counts())

landing_outcomes=data["Outcome"].value_counts()
print(landing_outcomes)

for i,outcome in enumerate(landing_outcomes.keys()):
  print(i,outcome)

bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)


'''








# Step 5: Prepare features and target variable
X = data.drop(['Outcome', 'FlightNumber', 'Date', 'Longitude', 'Latitude'], axis=1)  # Dropping unnecessary columns
y = data['Outcome']  

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model (Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


'''
