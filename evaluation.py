from datetime import datetime

import os
import numpy as np
import pandas as pd
import xgboost as xgb

from train import evaluate, load_model, predict


def main():
    print(f"Time Started: {datetime.now()}")

    model_path = "artifacts/model/xgboost_model.json"
    model = load_model(model_path)

    url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
    # load the data into pandas dataframe
    df = pd.read_csv(url)
    x = df.drop("Adopted", axis=1)

    # Extract categorical features
    cats = x.select_dtypes(exclude=np.number).columns.tolist()
    # Convert to Pandas category
    for col in cats:
        x[col] = x[col].astype("category")
    dx_mat = xgb.DMatrix(x, enable_categorical=True)

    y_preds_decoded = predict(model, dx_mat)
    df["Adopted Prediction"] = y_preds_decoded

    # Save the results to output directory
    if not os.path.exists("output"):
        os.mkdir("output")
    df.to_csv("output/results.csv", index=False)

    actual_values = df["Adopted"].values
    predicted_values = df["Adopted Prediction"].values
    print("\nEvaluation on all rows of dataset:\n")
    evaluate(actual_values, predicted_values)
    print(f"\nTime Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
