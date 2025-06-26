import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import kagglehub
import os

def train_model():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Load and prepare data
    path = kagglehub.dataset_download("tejashvi14/travel-insurance-prediction-data")
    data = pd.read_csv(os.path.join(path, "TravelInsurancePrediction.csv"))

    # Data preprocessing
    data = data.drop(columns=['Unnamed: 0'], axis=1)
    data['Employment Type'] = data['Employment Type'].replace({
        'Government Sector': 0,
        'Private Sector/Self Employed': 1
    })
    data = data.replace({'Yes': 1, 'No': 0})

    # Features and target
    features = data.drop(columns=['TravelInsurance'])
    target = data['TravelInsurance']

    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled_df, target, test_size=0.2, random_state=42
    )

    # Define and optimize Random Forest
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Best Random Forest Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)

    # Save the best model and scaler
    joblib.dump(best_model, os.path.join('model.pkl'))
    joblib.dump(scaler, os.path.join('scaler.pkl'))

    print("\nModel and scaler saved successfully!")

if __name__ == "__main__":
    train_model()
    