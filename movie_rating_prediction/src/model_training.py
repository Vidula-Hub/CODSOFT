from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_models(X_train, y_train):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save models
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(project_root, 'models')
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    return lr_model, rf_model