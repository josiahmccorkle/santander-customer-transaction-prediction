# ---------------------------------
#  Argument parsing
# ---------------------------------
# Initialize the parser
import argparse
import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from my_preprocessing import clean_data, preprocess


parser = argparse.ArgumentParser(description="passing in test flag to run predictions")
# Add a flag (boolean)
parser.add_argument("--test", action="store_true", help="Load and run predictions on test.csv instead of training data")
# Parse the arguments
args = parser.parse_args()

# --------------------------------
#  Load Data
# --------------------------------
# List all CSV files in a folder
filePath = "./data/"
files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
print(files)
# # # Load each file into a dictionary of DataFrames
expected_files = {"train.csv", "test.csv"}
dfs = {file: pd.read_csv(os.path.join(filePath, file)) for file in files if file in expected_files}

try:
    df = dfs["test.csv"] if args.test else dfs["train.csv"]
except KeyError as e:
    raise FileNotFoundError(f"Missing file: {e.args[0]} in './data/'")

# --------------------------------
# 5. Clean Data
# --------------------------------
df = clean_data(df)

# --------------------------------
# 6. Prepare Features and Target
# --------------------------------
if not args.test:
    y = df["target"]
    X = df.drop(["target", "ID_code"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X = df.drop(["ID_code"], axis=1)

# --------------------------------
# 7. Build Model Pipeline
# --------------------------------
print("Starting preprocessing...")
preprocessor = preprocess(X)
print("Preprocessing complete.")
# model_pipeline = Pipeline(steps=[("preprocess", preprocessor),("model", Ridge())])
RC_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42)
model_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RC_classifier)
])

# --------------------------------
# 8. Train or Predict
# --------------------------------
# Ensure outputs directory exists
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, "model.pkl")

if not args.test:
    print("Starting model training...")
    start = time.time()
    model_pipeline.fit(X_train, y_train)
    print("Training complete in", round(time.time() - start, 2), "seconds")
    print("Model training complete.")


    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    joblib.dump(model_pipeline, model_path)  # Save the trained model
    print("Scoring model...")
    print("Validation accuracy:", model_pipeline.score(X_test, y_test))
else:
    if not os.path.exists(model_path):
        raise FileNotFoundError("🚨 model.pkl not found. Please run training first to create the model.")
    model_pipeline = joblib.load(model_path)
    predictions = model_pipeline.predict(X)
    submission = pd.DataFrame({
        "ID_code": df["ID_code"],
        "target": predictions
    })
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print("Submission saved to submission.csv")
