import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Assignment 2 Classification App",
    layout="wide"
)

st.title("Machine Learning Assignment 2: Classification Model App")

st.write(
    "This Streamlit web application demonstrates multiple classification "
    "models trained on the Breast Cancer Wisconsin Diagnostic Dataset."
)

st.markdown("""
### App Features
- Upload CSV test data
- Select machine learning model from dropdown
- Display evaluation metrics
- Show confusion matrix
- Show classification report
- Display prediction output
""")

@st.cache_data
def load_default_test_data():
    return pd.read_csv("test_data.csv")


@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)


with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


metrics_df = pd.read_csv("model_metrics.csv")


model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}


st.sidebar.header("User Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Uploaded CSV file loaded successfully.")
else:
    data = load_default_test_data()
    st.info("Using default test_data.csv included in the repository.")


st.subheader("Preview of Test Data")
st.dataframe(data.head())


missing_features = [col for col in feature_names if col not in data.columns]

if missing_features:
    st.error(
        "The uploaded file is missing the following required feature columns: "
        + ", ".join(missing_features)
    )
    st.stop()


X_test_app = data[feature_names]

if "target" in data.columns:
    y_test_app = data["target"]
else:
    y_test_app = None


model = load_model(model_files[selected_model_name])

y_pred = model.predict(X_test_app)

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X_test_app)[:, 1]
else:
    y_score = model.decision_function(X_test_app)


st.subheader(f"Selected Model: {selected_model_name}")


if y_test_app is not None:

    metric_values = {
        "Accuracy": accuracy_score(y_test_app, y_pred),
        "AUC": roc_auc_score(y_test_app, y_score),
        "Precision": precision_score(y_test_app, y_pred, pos_label=1),
        "Recall": recall_score(y_test_app, y_pred, pos_label=1),
        "F1 Score": f1_score(y_test_app, y_pred, pos_label=1),
        "MCC": matthews_corrcoef(y_test_app, y_pred)
    }

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{metric_values['Accuracy']:.4f}")
    col2.metric("AUC", f"{metric_values['AUC']:.4f}")
    col3.metric("Precision", f"{metric_values['Precision']:.4f}")
    col4.metric("Recall", f"{metric_values['Recall']:.4f}")
    col5.metric("F1 Score", f"{metric_values['F1 Score']:.4f}")
    col6.metric("MCC", f"{metric_values['MCC']:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test_app, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model_name}")

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test_app,
        y_pred,
        target_names=["malignant", "benign"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

else:
    st.warning(
        "No target column found in uploaded CSV. "
        "Only predictions will be displayed."
    )


output_data = data.copy()
output_data["predicted_target"] = y_pred
output_data["predicted_class"] = output_data["predicted_target"].map({
    0: "malignant",
    1: "benign"
})

st.subheader("Prediction Output")
st.dataframe(output_data.head(20))


st.subheader("Overall Model Comparison From Notebook Experiment")
st.dataframe(metrics_df)


st.markdown("""
### Class Label Meaning
- `0` means malignant
- `1` means benign
""")


st.caption(
    "Developed for ML Assignment 2. Replace README links with your actual "
    "GitHub repository link and Streamlit deployment link before final PDF submission."
)
