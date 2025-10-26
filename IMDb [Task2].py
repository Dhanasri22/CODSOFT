import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
try:
    df = pd.read_csv("IMDb.csv", encoding='latin1')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: IMDb.csv file not found. Please ensure the file exists in the current directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print("\nAvailable columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

if 'Rating' not in df.columns:
    rating_alternatives = ['rating', 'RATING', 'IMDb Rating', 'imdb_rating', 'Score']
    for alt in rating_alternatives:
        if alt in df.columns:
            df.rename(columns={alt: 'Rating'}, inplace=True)
            print(f"\nRenamed '{alt}' to 'Rating'")
            break
    else:
        print("\nError: No rating column found in dataset!")
        exit()
possible_features = {
    'Genre': ['Genre', 'genre', 'GENRE', 'Genres'],
    'Director': ['Director', 'director', 'DIRECTOR'],
    'Actors': ['Actors', 'actors', 'ACTORS', 'Actor', 'Cast'],
    'Year': ['Year', 'year', 'YEAR', 'Release Year'],
    'Duration': ['Duration', 'duration', 'Runtime', 'runtime']
}
column_mapping = {}
for standard_name, alternatives in possible_features.items():
    for alt in alternatives:
        if alt in df.columns:
            column_mapping[alt] = standard_name
            break

df.rename(columns=column_mapping, inplace=True)
print("\nStandardized column names:", df.columns.tolist())

df.dropna(subset=['Rating'], inplace=True)

if 'Title' in df.columns:
    df.drop(['Title'], axis=1, inplace=True)

categorical_cols = ['Genre', 'Director', 'Actors']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

numerical_cols = ['Year', 'Duration']
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

df.dropna(inplace=True)

print(f"\nDataset shape after cleaning: {df.shape}")

if len(df) == 0:
    print("Error: No data remaining after cleaning!")
    exit()

X = df.drop('Rating', axis=1)
y = df['Rating']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

categorical_features = [col for col in ['Genre', 'Director', 'Actors'] if col in X.columns]
numeric_features = [col for col in X.columns if col not in categorical_features]

print(f"\nCategorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")

transformers = []

if categorical_features:
    transformers.append(
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    )

if numeric_features:
    transformers.append(
        ('num', 'passthrough', numeric_features)
    )

if not transformers:
    print("Error: No features available for training!")
    exit()

preprocessor = ColumnTransformer(transformers=transformers)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

if len(X) < 10:
    print("Error: Not enough data for train-test split!")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

print("\nTraining model...")
try:
    model_pipeline.fit(X_train, y_train)
    print("Model trained successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    exit()

print("\nMaking predictions...")
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print("="*50)

print("\nSample Predictions (first 10):")
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_pred[:10],
    'Difference': y_test.iloc[:10].values - y_pred[:10]
})
print(comparison_df.to_string(index=False))

if hasattr(model_pipeline.named_steps['regressor'], 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    feature_names = []
    
    if categorical_features:
        cat_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))
    
    if numeric_features:
        feature_names.extend(numeric_features)
    
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    print(feature_importance_df.to_string(index=False))