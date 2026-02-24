import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def load_data(data_path, columns_path):
    with open(columns_path) as f:
        columns = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(data_path, names=columns)
    if 'hispanic origin' in df.columns:
        df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')
    return df

def clean_categorical_ints(df):
    df = df[df['age'] >= 18]
    df = df.replace('?', 'Unknown')
    return df

def engineer_features(df):
    df['log_capital_gains'] = np.log1p(df['capital gains'])
    df['log_dividends']     = np.log1p(df['dividends from stocks'])
    df['has_investments']   = (
        (df['capital gains'] > 0) |
        (df['dividends from stocks'] > 0) |
        (df['capital losses'] > 0)
    ).astype(int)
    df['log_wage_per_hour'] = np.log1p(df['wage per hour'])

    return df

def select_features(df):
    features = [
        # Core demographic
        "age",
        "sex",
        "marital stat",
        "race",

        # Employment
        "weeks worked in year",
        "full or part time employment stat",
        "class of worker",
        "major occupation code",
        "major industry code",
        "num persons worked for employer",

        # Education
        "education",

        # Financial
        "tax filer stat",
        "log_wage_per_hour",
        "log_capital_gains",
        "log_dividends",
        "has_investments",
    ]
    return df[features]

def preprocess_for_clustering(X, y):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", TargetEncoder(random_state=RANDOM_STATE), categorical_cols)
    ])
    X_processed = preprocessor.fit_transform(X, y)
    return X_processed, preprocessor

def reduce_dimensionality(X, variance_threshold=0.9):
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced to {X_pca.shape[1]} components (explained variance: {variance_threshold*100:.0f}%)")
    return X_pca, pca

def run_kmeans(X, sample_weights, k_range=(2, 10)):
    best_k = None
    best_score = -1
    scores = []
    for k in range(*k_range):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = kmeans.fit_predict(X, sample_weight=sample_weights)
        sample_idx = np.random.choice(len(X), size=min(20000, len(X)), replace=False)
        score = silhouette_score(X[sample_idx], labels[sample_idx], random_state=RANDOM_STATE)
        scores.append((k, score))
        print(f'K={k}, Silhouette={score:.4f}')
        if score > best_score:
            best_score = score
            best_k     = k
    print(f'Best K by silhouette: {best_k}')
    return best_k, scores

def cluster_and_profile(df, X_pca, k, sample_weights):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_pca, sample_weight=sample_weights)

    num_profile = df.groupby('cluster').agg({
        "age":                             "mean",
        "weeks worked in year":            "mean",
        "wage per hour":                   "mean",
        "capital gains":                   "mean",
        "dividends from stocks":           "mean",
        "num persons worked for employer": "mean",
        "label_binary":                    "mean"
    }).round(2)

    for col in ["education", "major occupation code", "marital stat", "sex"]:
        if col in df.columns:
            num_profile[col + "_mode"] = df.groupby("cluster")[col].agg(
                lambda x: x.mode().iloc[0]
            )

    print(num_profile.to_string())
    return df, num_profile

def plot_clusters(X_pca, labels):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=10)
    plt.title('PCA-reduced Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.show()

def plot_silhouette_scores(scores):
    k_values, silhouette_scores = zip(*scores)
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=k_values, y=silhouette_scores, marker='o')
    plt.title('Silhouette Scores for KMeans')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.show()

if __name__ == "__main__":
    DATA_PATH    = "../data/census-bureau.data"
    COLUMNS_PATH = "../data/census-bureau.columns"

    df = load_data(DATA_PATH, COLUMNS_PATH)
    df = clean_categorical_ints(df)
    df['label_binary'] = df['label'].apply(lambda x: 1 if '50000+' in x else 0)
    df = engineer_features(df)

    X = select_features(df)
    X_processed, preprocessor = preprocess_for_clustering(X, df['label_binary'])
    X_pca, pca = reduce_dimensionality(X_processed, variance_threshold=0.9)

    sample_weights = df['weight'].values

    best_k, scores = run_kmeans(X_pca, sample_weights, k_range=(2, 10))
    df, profile = cluster_and_profile(df, X_pca, 5, sample_weights) # hardcoded to 5 for more logical clusters
    plot_clusters(X_pca, df['cluster'])
    plot_silhouette_scores(scores)