import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.3

def clean_categorical_ints(df):
    df = df.replace('?', 'Unknown')

    # Integer-coded categorical columns
    cat_int_cols = [
        'detailed industry recode',
        'detailed occupation recode',
        "own business or self employed",
        "veterans benefits",
    ]

    # Convert to categorical (string so TargetEncoder handles them)
    for col in cat_int_cols:
        df[col] = df[col].astype(str)

    return df

def engineer_features(df):
    # Work intensity
    df["annual_labor_proxy"] = df["weeks worked in year"] * df["wage per hour"]
    df["full_year_worker"] = (df["weeks worked in year"] >= 50).astype(int)
    # Capital income indicators
    df["has_capital_gains"] = (df["capital gains"] > 0).astype(int)
    df["has_dividends"] = (df["dividends from stocks"] > 0).astype(int)
    df["net_capital"] = df["capital gains"] - df["capital losses"]
    return df

def select_features(df):
    selected_cols = [
        # Original high-importance numeric
        "age",
        "weeks worked in year",
        "dividends from stocks",
        "num persons worked for employer",
        # Original categorical
        "education",
        "sex",
        "tax filer stat",
        "detailed occupation recode",
        "detailed industry recode",
        "class of worker",
        # Engineered
        "annual_labor_proxy",
        "full_year_worker",
        "has_capital_gains",
        "has_dividends",
        "net_capital",
    ]
    return df[selected_cols]

def load_data(data_path, columns_path):
    with open(columns_path) as f:
        columns = [line.strip() for line in f if line.strip()]
    df = pd.read_csv(data_path, names=columns)
    return df

def preprocess_data(df):
    """Preprocess data: feature engineering, select features, encode, split train/test."""
    df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')
    df['label_binary'] = df['label'].apply(lambda x: 1 if '50000+' in x else 0)
    df = engineer_features(df)

    X = select_features(df)
    y = df['label_binary']
    weights = df['weight']

    # Now get column types from X only
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'str']).columns.tolist()

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", TargetEncoder(random_state=RANDOM_STATE), categorical_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    return X_train_processed, X_test_processed, y_train, y_test, w_train, w_test, feature_names

def train_model(X_train, y_train, w_train):
    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=8,
        n_estimators=200,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model

def evaluate_model(model, X_test, y_test, w_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
    print("\nEvaluation Metrics (Test Set):")
    print(f"ROC-AUC:   {roc_auc:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, sample_weight=w_test, digits=3))
    return roc_auc

if __name__ == "__main__":
    DATA_PATH = "../data/census-bureau.data"
    COLUMNS_PATH = "../data/census-bureau.columns"
    print("Loading data...")
    df = load_data(DATA_PATH, COLUMNS_PATH)
    df = clean_categorical_ints(df)
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, w_train, w_test, feature_names = preprocess_data(df)
    print("Training XGBoost Model...")
    model = train_model(X_train, y_train, w_train)
    print("Evaluating model...")
    full_auc = evaluate_model(model, X_test, y_test, w_test)