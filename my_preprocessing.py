from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor

def clean_data(df, drop_thresh=0.9):
    # Drop columns with too many missing values based on a threshold (default: >90% missing)
    missing = df.isnull().sum() / len(df)
    # Identify which columns are being dropped and print them for logging/debugging
    dropped = missing[missing > drop_thresh].index
    print(f"Dropping columns: {list(dropped)}")

    # Drop columns with too many missing values
    df = df.drop(columns=dropped)

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "int64"]:
                # For numeric columns, fill missing values with the column median
                df[col] = df[col].fillna(df[col].median())
            else:
                # For categorical columns, fill missing values with "None" (or most frequent if using ColumnTransformer)
                df[col] = df[col].fillna("None")

    # Note: This function does not handle encoding or scaling — those will be added in preprocess()
    return df
