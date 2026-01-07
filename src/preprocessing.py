import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load Dataset Function
path = r"C:\Users\shiva\OneDrive\Desktop\customer_churn\data\raw\bank_customer_churn.csv"

def load_data(path):
    """
    Load dataset from given path
    """
    return pd.read_csv(path)


# Separate Features & Target
def split_features_target(df, target_column="Exited"):
    """
    Separate features (X) and target (y)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# Identify Column Types
def get_column_types(X):
    """
    Identify numerical and categorical columns
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical_cols, numerical_cols


# Build preprocessing pipeline
def build_preprocessor(categorical_cols, numerical_cols):
    """
    Create preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor


# Train-Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# Test preprocessing
if __name__ == "__main__":
    df = load_data(path)
    X, y = split_features_target(df)
    cat_cols, num_cols = get_column_types(X)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Categorical Columns:", cat_cols)
    print("Numerical Columns:", num_cols)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
