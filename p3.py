#Model training
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle


data = pd.read_csv('SpaceX_Falcon9.csv')
print(data)

data['PayloadMass'] = data['PayloadMass'].fillna(data['PayloadMass'].median())
data['Outcome'] = data['Outcome'].fillna(data['Outcome'].mode()[0])
data['BoosterVersion'] = data['BoosterVersion'].fillna(data['BoosterVersion'].mode()[0])
data['LaunchSite'] = data['LaunchSite'].fillna(data['LaunchSite'].mode()[0])
data['Orbit'] = data['Orbit'].fillna(data['Orbit'].mode()[0])
data['LandingPad'] = data['LandingPad'].fillna(data['LandingPad'].mode()[0])
data['Block'] = data['Block'].fillna(data['Block'].mode()[0])
data['Serial'] = data['Serial'].fillna(data['Serial'].mode()[0])



print(data.shape)

print(data.head(100))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y, y_predict):
    """This function plots the confusion matrix"""
    
    # Generate confusion matrix
    cm = confusion_matrix(y, y_predict)
    
    # Create subplot
    ax = plt.subplot()
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues") 
    
    # Set labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # Set tick labels for axes
    ax.xaxis.set_ticklabels(['Did not land', 'Land'])
    ax.yaxis.set_ticklabels(['Did not land', 'Land'])
    
    # Show the plot
    plt.show()
y = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Sample predicted labels (y_predict)
y_predict = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]

# Call the function
plot_confusion_matrix(y, y_predict)


Y = data["Outcome"].to_numpy()
print(type(Y))
print(Y[:5])

le = LabelEncoder()

data['Outcome'] = le.fit_transform(data['Outcome'])
data['BoosterVersion'] = le.fit_transform(data['BoosterVersion'])
data['LaunchSite'] = le.fit_transform(data['LaunchSite'])
data['Orbit'] = le.fit_transform(data['Orbit'])
data['LandingPad'] = le.fit_transform(data['LandingPad'])
data['Block'] = le.fit_transform(data['Block'])
data['Serial'] = le.fit_transform(data['Serial'])
data['Date']=le.fit_transform(data['Date'])
data['FlightNumber']=le.fit_transform(data['FlightNumber'])

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = ['FlightNumber', 'BoosterVersion', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial', 'Longitude', 'Latitude']
#X = data[features]
y = data['Outcome']

X = scaler.fit_transform(features)
'''

from sklearn.preprocessing import StandardScaler
features = ['FlightNumber', 'BoosterVersion', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial', 'Longitude', 'Latitude']

# Selecting the actual data (not the column names) for scaling
X = data[features]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)






print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=20)
X_train, X_val, y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=20)

print(f"Training Set Size : {X_train.shape}")
print(f"Validation Set Size : {X_val.shape}")
print(f"Testing Set Size : {X_test.shape}")

print(y_test.shape)

#Logistic Regression
lr=LogisticRegression(random_state=20)
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

def plot_confusion_matrix(y, y_predict):
    """This function plots the confusion matrix"""
    cm = confusion_matrix(y, y_predict)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not land', 'Landed'], yticklabels=['Did not land', 'Landed'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Compute training and test accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy_lr = accuracy_score(y_test, y_test_pred)

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy_lr:.4f}")
plot_confusion_matrix(y_test, y_pred)


#SVM
def plot_confusion_matrix(y, y_predict):
    """This function plots the confusion matrix"""
    cm = confusion_matrix(y, y_predict)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not land', 'Landed'], yticklabels=['Did not land', 'Landed'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Initialize and train SVM model
svm_model = SVC(kernel='linear', C=1, random_state=42)  # Using a linear kernel
svm_model.fit(X_train, y_train)

y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

# Compute training and test accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy_svm = accuracy_score(y_test, y_test_pred)

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy_svm:.4f}")

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix for test set
plot_confusion_matrix(y_test, y_test_pred)






#Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Training and Testing Accuracy
print("Decision Tree Classifier")
print(f"Training Accuracy: {accuracy_score(y_train, dt_model.predict(X_train)):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Compute training and test accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

test_accuracy_dt= accuracy_score(y_test, y_test_pred)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()




#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can tune 'n_neighbors'
knn_model.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_model.predict(X_test)

# Training and Testing Accuracy
print("K-Nearest Neighbors (KNN) Classifier")
print(f"Training Accuracy: {accuracy_score(y_train, knn_model.predict(X_train)):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
test_accuracy_knn = accuracy_score(y_test, y_test_pred)

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN")
plt.show()




#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Training and Testing Accuracy
print("Random Forest Classifier")
print(f"Training Accuracy: {accuracy_score(y_train, rf_model.predict(X_train)):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

test_accuracy_rf= accuracy_score(y_test, y_test_pred)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()




#finding best model

# Create a dictionary to store the test accuracy for each model
model_accuracies = {
    "Logistic Regression": test_accuracy_lr,
    "SVM": test_accuracy_svm,
    "Decision Tree":test_accuracy_dt,
    "KNN":test_accuracy_knn,
    "Random Forest":test_accuracy_rf
}

# Print all model accuracies
for model, accuracy in model_accuracies.items():
    print(f"{model} Test Accuracy: {accuracy:.4f}")

# Find the best model based on test accuracy
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"\n🏆 Best Model: {best_model} with Test Accuracy: {model_accuracies[best_model]:.4f}")





#save the trained model
with open('lr_model.pkl', 'wb') as file:
    pickle.dump(dt_model, file)

print("Model saved to 'dt_model.pkl'")