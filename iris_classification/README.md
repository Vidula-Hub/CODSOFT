# Iris Flower Classification using Machine Learning

This project uses the Iris dataset to classify flowers into Setosa, Versicolor, or Virginica using machine learning models (Decision Tree, SVM, KNN).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Video Demo](#video-demo)
- [License](#license)

## Overview
The goal is to train and evaluate classification models using features like sepal length, sepal width, petal length, and petal width.

## Dataset
The Iris dataset from `sklearn.datasets` contains:
- 150 samples
- 4 features: sepal length, sepal width, petal length, petal width
- 3 classes: Setosa, Versicolor, Virginica

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris_classification.git

Set up a virtual environment:
python -m venv venv
.\venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Usage
Run EDA:
jupyter notebook notebooks/iris_eda.ipynb
Preprocess data:
python src/data_preprocessing.py
Train models:
python src/model_training.py
Evaluate models:
python src/evaluation.py
Results
Decision Tree Accuracy: ~0.97
SVM Accuracy: ~0.98
KNN Accuracy: ~0.97