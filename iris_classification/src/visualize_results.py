from sklearn.datasets import load_iris
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure images folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Load Iris dataset for feature names
iris = load_iris()
feature_names = iris.feature_names

# Load the trained Decision Tree model
dt_model = joblib.load('models/decision_tree_model.pkl')

# Feature importance for Decision Tree
importances = dt_model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Decision Tree Feature Importance')
plt.savefig('images/feature_importance.png')
plt.show()

print("Feature importance plot saved to images/feature_importance.png")