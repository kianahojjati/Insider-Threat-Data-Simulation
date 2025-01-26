import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Example preprocessing
    data['normalized_login_time'] = StandardScaler().fit_transform(data[['login_time']])
    data['encoded_role'] = LabelEncoder().fit_transform(data['role'])
    data.fillna(0, inplace=True)  # Handle missing values
    
    return data

# 2. Feature Engineering
def extract_features(data):
    # Example feature engineering
    data['working_hours'] = data['logout_time'] - data['login_time']
    data['suspicious_access'] = (data['file_access_count'] > 10).astype(int)
    
    # Dropping irrelevant columns (customize based on your dataset)
    data = data.drop(['user_id', 'role'], axis=1)
    
    return data

# 3. Model Training
def train_model(data):
    X = data.drop('threat_label', axis=1)
    y = data['threat_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Plotting feature importance
    feature_importance = model.feature_importances_
    plt.barh(X_test.columns, feature_importance)
    plt.title("Feature Importance")
    plt.show()

# 5. Main Execution
if __name__ == "__main__":
    # Load the data
    print("Loading and preprocessing data...")
    file_path = "path/to/your/dataset.csv"  # Update with your dataset path
    data = load_and_preprocess_data(file_path)
    
    # Feature engineering
    print("Extracting features...")
    data = extract_features(data)
    
    # Train the model
    print("Training the model...")
    model, X_test, y_test = train_model(data)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    
    print("Simulation complete!")
