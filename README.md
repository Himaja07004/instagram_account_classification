## Instagram Account Classification
This project builds a machine learning model to classify Instagram accounts as fake or genuine based on various features extracted from account data.

## Features
- Data loading and preprocessing
- Data visualization including:
- Distribution of fake vs genuine accounts
- Correlation heatmap of features
- Profile picture presence in fake vs genuine accounts
- Followers count comparison
- Feature scaling using StandardScaler
- Splitting dataset into training and testing sets
- Training a Random Forest classifier
- Model evaluation using classification report and confusion matrix
- Feature importance visualization

## Installations
- Create a python virtual environment:
  python -m venv venv
  .\venv\Scripts\activate
- install required libraries
  pip install pandas numpy matplotlib seaborn scikit-learn

## Usage
- Place Instagram account dataset in CSV format and update the filename in the script (instagramdata.csv by default).
- Run the script to train the model and see the classification results and visualizations:
  python account_classification.py
