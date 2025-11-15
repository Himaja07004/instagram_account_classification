import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'instagramdata.csv' with your dataset path)
data = pd.read_csv('instagramdata.csv')

# Display first few rows
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Basic statistics
print(data.describe())

# Distribution of target variable
sns.countplot(x='fake', data=data)
plt.title('Distribution of Fake (1) vs Genuine (0) Accounts')
print(data.columns)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
print(data.columns)
plt.show()

# Profile picture presence vs fake
sns.barplot(x='fake', y='profile pic', data=data)
plt.title('Profile Picture Presence in Fake vs Genuine Accounts')
print(data.columns)
plt.show()

# Followers count comparison
sns.boxplot(x='fake', y='#followers', data=data)
plt.title('Followers Count in Fake vs Genuine Accounts')
print(data.columns)
plt.show()
# Features and target
X = data.drop('fake', axis=1)
y = data['fake']

#  Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to DataFrame for convenience
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)
# Predictions on test set
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine', 'Fake']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(y=X.columns[indices], x=importances[indices], palette='viridis')
plt.title('Feature Importances')
plt.show()

