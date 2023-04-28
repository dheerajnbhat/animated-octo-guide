from datetime import datetime
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def train_test_val_split(df: pd.core.frame.DataFrame):
    """
    Function to split the dataframe into train, validation and test splits
    Parameters
    ----------
    df: pd.core.frame.DataFrame
        Pandas DataFrame

    Returns
    -------
    x_train: pd.core.frame.DataFrame
        Training features
    x_val: pd.core.frame.DataFrame
        Validation features
    x_test: pd.core.frame.DataFrame
        Test  features
    y_train: np.ndarray
        Training target values
    y_val: np.ndarray
        Validation  target values
    y_test: np.ndarray
        Test  target values
    """
    # create input features and target feature
    x, y = df.drop("Adopted", axis=1), df[["Adopted"]]

    ordinal_enc = OrdinalEncoder().fit(y)
    y_encoded = ordinal_enc.transform(y)

    # Extract categorical features
    cats = x.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        x[col] = x[col].astype("category")

    # split the dataset into train and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42
    )
    # split the train dataset into train and validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    print("=" * 20)
    print("Dataset INFO:")
    print("=" * 20)
    print(f"Shape of Training set: {x_train.shape}")
    print(f"Shape of Validation set: {x_val.shape}")
    print(f"Shape of Testing set: {x_test.shape}")
    print(
        f"Ratio of Train: Val: Test = {len(x_train)/len(x):.03}: {len(x_val)/len(x):.03}: {len(x_test)/len(x):.03}"
    )
    print("=" * 20)
    return x_train, x_val, x_test, y_train, y_val, y_test


def get_dmatrix(df):
    """
    Convert dataframe to DMatrix

    Parameters
    ----------
    df: pd.core.frame.DataFrame
        input features dataframe

    Returns
    -------
    DMatrix
    """
    # Extract categorical features
    cats = df.select_dtypes(exclude=np.number).columns.tolist()
    # Convert to Pandas category
    for col in cats:
        df[col] = df[col].astype("category")
    dx_mat = xgb.DMatrix(df, enable_categorical=True)
    return dx_mat


def load_model(model_path: str):
    """
    Parameters
    ----------
    model_path: str
        path to model

    Returns
    -------
    model: xgb.core.Booster
        XGBoost model loaded from model path
    """
    # Load the model from the artifacts folder
    model = xgb.Booster({"nthread": 0})
    model.load_model(model_path)
    return model


def predict(model: xgb.core.Booster, d_mat: xgb.core.DMatrix):
    """
    Parameters
    ----------
    model: xgb.core.Booster
        XGBoost model
    d_mat: xgb.core.DMatrix
        Dmatrix of the data

    Returns
    -------
    y_preds_decoded: list(int)
        Predictions for data samples
    """
    target_dict = {0: "No", 1: "Yes"}
    y_preds = model.predict(d_mat)
    y_preds = [round(i) for i in y_preds]
    y_preds_decoded = [target_dict[i] for i in y_preds]
    return y_preds_decoded


def evaluate(actual_values: list[str], predicted_values: list[str]):
    """
    Parameters
    ----------
    actual_values: list(str)
        True values
    predicted_values: list(str)
        Predicted values from the model
    """
    # Classification report containing F1, accuracy, recall and precision score
    clf_report = classification_report(actual_values, predicted_values)
    print(f"Classification Report:\n\n{clf_report}")
    # Confusion matrix
    cm = confusion_matrix(actual_values, predicted_values, labels=["No", "Yes"])
    print(f"\n\nConfusion Matrix:\n\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/charts/confusion_matrix.png")
    # AUC
    rev_td = {"No": 0, "Yes": 1}
    fpr, tpr, thresholds = roc_curve(
        [rev_td[i] for i in actual_values], [rev_td[i] for i in predicted_values]
    )
    roc_score = roc_auc_score(
        [rev_td[i] for i in actual_values], [rev_td[i] for i in predicted_values]
    )
    plt.subplots(1, figsize=(10, 10))
    plt.title("ROC score: {0:0.4f}".format(roc_score))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.plot(fpr, tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig("artifacts/charts/roc_curve.png")


def main():
    print(f"Time Started: {datetime.now()}")
    url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"

    # load the data into pandas dataframe
    df = pd.read_csv(url)
    # Split the dataset into train, validation and test set
    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(df)

    # Create classification matrices for train, val and test set
    dtrain_mat = xgb.DMatrix(x_train, y_train, enable_categorical=True)
    dval_mat = xgb.DMatrix(x_val, y_val, enable_categorical=True)
    dtest_mat = xgb.DMatrix(x_test, enable_categorical=True)

    # Define hyperparameters in a dictionary
    yes_count = df["Adopted"].value_counts()["Yes"]
    no_count = df["Adopted"].value_counts()["No"]
    scale_weight = no_count / yes_count

    params = {
        "objective": "binary:logistic",
        "tree_method": "auto",
        "scale_pos_weight": scale_weight,
    }
    evals = [(dval_mat, "validation")]

    num_rounds = 100
    print(f"Training")
    model = xgb.train(
        params=params,
        dtrain=dtrain_mat,
        num_boost_round=num_rounds,
        evals=evals,
        verbose_eval=2,
        early_stopping_rounds=5,
    )

    # Save the model in the artifacts directory
    model_path = "artifacts/model/xgboost_model.json"
    model.save_model(model_path)
    print(f"\nModel saved at: {model_path}\n")
    # Evaluating Test dataset
    y_preds = predict(model, dtest_mat)
    target_dict = {0: "No", 1: "Yes"}
    y_test = [target_dict[i[0]] for i in y_test]
    print("\nEvaluation on only Test split of dataset:\n")
    evaluate(y_test, y_preds)
    print(f"\nTime Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
