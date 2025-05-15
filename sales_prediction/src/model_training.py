# src/model_training.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # Use absolute path
    project_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from src/
    models_dir = os.path.join(project_root, 'models')
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    return lr_model, rf_model