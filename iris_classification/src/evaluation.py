from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Ensure images folder exists
if not os.path.exists('images'):
    os.makedirs('images')

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load models
dt_model = joblib.load('models/decision_tree_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

# Evaluate models
models = {'Decision Tree': dt_model, 'SVM': svm_model, 'KNN': knn_model}
for name, model in models.items():
    # Predict
    X_eval = X_test if name == 'Decision Tree' else X_test_scaled
    y_pred = model.predict(X_eval)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.2f}")
    
    # Classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'images/{name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()