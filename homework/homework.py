import pandas as pd
import os
import gzip
import pickle
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression="zip")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.loc[df["MARRIAGE"] != 0] 
    df = df.loc[df["EDUCATION"] != 0] 
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return df

def create_pipeline() -> Pipeline:
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

def create_estimator(pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

def save_model(path: str, estimator: GridSearchCV):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def calculate_precision_metrics(dataset_name: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def calculate_confusion_metrics(dataset_name: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

def main():
    input_files_path = "files/input/"
    models_files_path = "files/models/"
    output_files_path = "files/output/"

    test_df = load_dataset(os.path.join(input_files_path, "test_data.csv.zip"))
    train_df = load_dataset(os.path.join(input_files_path, "train_data.csv.zip"))

    test_df = clean_dataset(test_df)
    train_df = clean_dataset(train_df)

    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    pipeline = create_pipeline()

    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    save_model(os.path.join(models_files_path, "model.pkl.gz"), estimator)

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calculate_precision_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calculate_precision_metrics("train", y_train, y_train_pred)

    test_confusion_metrics = calculate_confusion_metrics("test", y_test, y_test_pred)
    train_confusion_metrics = calculate_confusion_metrics("train", y_train, y_train_pred)

    os.makedirs(output_files_path, exist_ok=True)
    with open(os.path.join(output_files_path, "metrics.json"), "w") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")

if __name__ == "__main__":
    main()