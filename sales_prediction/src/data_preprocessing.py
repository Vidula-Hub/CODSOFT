import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.mean(), inplace=True)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler