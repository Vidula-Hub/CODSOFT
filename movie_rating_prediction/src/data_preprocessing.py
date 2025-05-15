# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    # Load dataset with latin1 encoding
    df = pd.read_csv(data_path, encoding='latin1')

    # Select relevant columns
    columns = ['Rating', 'Genre', 'Duration', 'Year', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    df = df[columns]

    # Handle missing values
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(float)
    df['Duration'] = df['Duration'].fillna(df['Duration'].median())
    df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)
    df['Year'] = df['Year'].fillna(df['Year'].median())
    df['Genre'] = df['Genre'].fillna('Unknown')
    df['Director'] = df['Director'].fillna('Unknown')
    df['Actor 1'] = df['Actor 1'].fillna('Unknown')
    df['Actor 2'] = df['Actor 2'].fillna('Unknown')
    df['Actor 3'] = df['Actor 3'].fillna('Unknown')

    # Encode categorical variables
    # Create dummy variables for genres (multi-label)
    df['Genre'] = df['Genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else ['Unknown'])
    genres = set()
    for genre_list in df['Genre']:
        genres.update(genre_list)
    for genre in genres:
        df[genre] = df['Genre'].apply(lambda x: 1 if genre in x else 0)

    # Frequency encoding for Director and Actors
    director_freq = df['Director'].value_counts().to_dict()
    df['director_encoded'] = df['Director'].map(director_freq)

    # Combine actors into a single column for frequency encoding
    df['actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda x: ','.join(x), axis=1)
    actors_freq = df['actors'].str.split(',', expand=True).stack().value_counts().to_dict()
    df['actors_encoded'] = df['actors'].apply(
        lambda x: sum([actors_freq.get(a, 0) for a in x.split(',')]) / len(x.split(',')) if x != 'Unknown' else 0
    )

    # Features and target
    genre_columns = list(genres)
    X = df[['Year', 'Duration', 'director_encoded', 'actors_encoded'] + genre_columns]
    y = df['Rating']

    # Convert numerical columns to float to avoid dtype mismatch
    X = X.astype({'Year': 'float', 'Duration': 'float', 'director_encoded': 'float', 'actors_encoded': 'float'})

    # Scale numerical features
    scaler = StandardScaler()
    X.loc[:, ['Year', 'Duration', 'director_encoded', 'actors_encoded']] = scaler.fit_transform(
        X[['Year', 'Duration', 'director_encoded', 'actors_encoded']]
    )

    return X, y, scaler