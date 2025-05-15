# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Path to the dataset
    data_path = 'data/IMDb Movies India.csv'
    print("Current Working Directory:", os.getcwd())
    print("Dataset path exists:", os.path.exists(data_path))

    # Preprocess data
    X, y, scaler = preprocess_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr_model, rf_model = train_models(X_train, y_train)

    # Evaluate models
    print("Evaluating Models...")
    lr_metrics = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
    rf_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

    print("\nLinear Regression Metrics:")
    for key, value in lr_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nRandom Forest Metrics:")
    for key, value in rf_metrics.items():
        print(f"{key}: {value:.4f}")

    # Visualize predictions (Random Forest)
    y_test_pred = rf_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings (Random Forest)')
    plt.savefig('images/actual_vs_predicted.png')
    plt.close()

    # Visualize feature importance (Random Forest)
    feature_names = X.columns
    importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importance (Random Forest)')
    plt.savefig('images/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()