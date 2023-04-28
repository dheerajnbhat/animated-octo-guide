import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

from train import get_dmatrix, load_model, predict

st.set_page_config(layout="wide")


@st.cache_resource
def st_load_model():
    model_path = "artifacts/model/xgboost_model.json"
    model = load_model(model_path)
    return model


def present():
    model = st_load_model()

    with st.sidebar:
        radio_opt = st.radio(
            "Choose a prediction type",
            ("Predict for Single Input", "Predict for dataframe"),
        )

    if radio_opt == "Predict for Single Input":
        st.text("Enter the values for feature:")
        try:
            input_form1 = st.form(key="input-form1")
            x_type = input_form1.text_input("Type", value="Dog")
            x_age = input_form1.text_input("Age", value="1")
            x_breed1 = input_form1.text_input("Breed1", value="Mixed Breed")
            x_gender = input_form1.text_input("Gender", value="Male")
            x_color1 = input_form1.text_input("Color1", value="Brown")
            x_color2 = input_form1.text_input("Color2", value="No Color")
            x_matsize = input_form1.text_input("MaturitySize", value="Medium")
            x_furlen = input_form1.text_input("FurLength", value="Short")
            x_vacc = input_form1.text_input("Vaccinated", value="No")
            x_sterl = input_form1.text_input("Sterilized", value="No")
            x_health = input_form1.text_input("Health", value="Healthy")
            x_fee = input_form1.text_input("Fee", value="0")
            x_photoamt = input_form1.text_input("PhotoAmt", value="1")
            input_submit1 = input_form1.form_submit_button("Submit")
            if input_submit1:
                row = [
                    [
                        x_type,
                        int(x_age),
                        x_breed1,
                        x_gender,
                        x_color1,
                        x_color2,
                        x_matsize,
                        x_furlen,
                        x_vacc,
                        x_sterl,
                        x_health,
                        int(x_fee),
                        int(x_photoamt),
                    ]
                ]
                cols = [
                    "Type",
                    "Age",
                    "Breed1",
                    "Gender",
                    "Color1",
                    "Color2",
                    "MaturitySize",
                    "FurLength",
                    "Vaccinated",
                    "Sterilized",
                    "Health",
                    "Fee",
                    "PhotoAmt",
                ]
                df = pd.DataFrame(data=row, columns=cols)
                dx_mat = get_dmatrix(df)
                result = predict(model, dx_mat)
                st.subheader(f"Adopted: {result[0]}")
            else:
                st.subheader("Please submit the input data!")
        except:
            st.subheader("Please enter the input features correctly!")

    elif radio_opt == "Predict for dataframe":
        input_form2 = st.form(key="input-form1")
        url = input_form2.text_input(
            "Enter the URL",
            value="gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv",
        )
        input_submit2 = input_form2.form_submit_button("Submit")
        if input_submit2:
            df = pd.read_csv(url)
            x = df.drop("Adopted", axis=1)
            dx_mat = get_dmatrix(x)
            y_preds = predict(model, dx_mat)
            df["Adopted Prediction"] = y_preds
            actual_values = df["Adopted"].values
            predicted_values = df["Adopted Prediction"].values
            clf_report = classification_report(
                actual_values, predicted_values, output_dict=True
            )
            st.subheader("Classification Report:")
            st.table(clf_report)
            col1, col2 = st.columns(2)
            with col1:
                plt.subplots(1, figsize=(1, 1))
                cm = confusion_matrix(
                    actual_values, predicted_values, labels=["No", "Yes"]
                )
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm, display_labels=["No", "Yes"]
                )
                disp.plot()
                st.subheader("Confusion Matrix:")
                st.pyplot(plt.gcf())
            with col2:
                # AUC
                rev_td = {"No": 0, "Yes": 1}
                fpr, tpr, thresholds = roc_curve(
                    [rev_td[i] for i in actual_values],
                    [rev_td[i] for i in predicted_values],
                )
                roc_score = roc_auc_score(
                    [rev_td[i] for i in actual_values],
                    [rev_td[i] for i in predicted_values],
                )
                plt.subplots(1, figsize=(3, 3))
                plt.title("ROC score: {0:0.4f}".format(roc_score))
                plt.plot([0, 1], ls="--")
                # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
                plt.plot(fpr, tpr)
                plt.ylabel("True Positive Rate")
                plt.xlabel("False Positive Rate")
                st.subheader("ROC Curve:")
                st.pyplot(plt.gcf())


if __name__ == "__main__":
    present()
