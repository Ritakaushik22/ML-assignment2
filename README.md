# Machine Learning Assignment 2: Classification Models and Streamlit App

## a. Problem Statement
The objective of this assignment is to build and compare multiple classification models on a public classification dataset and deploy the trained models using an interactive Streamlit web application. The application allows a user to upload test data, select a model, view predictions, and evaluate classification performance using standard metrics.

## b. Dataset Description
The dataset used in this project is the Breast Cancer Wisconsin Diagnostic Dataset. It is a binary classification dataset originally available from the UCI Machine Learning Repository and also available through scikit-learn.

- Number of instances: 569
- Number of features: 30
- Target classes: malignant and benign
- Target encoding used in this project: 0 = malignant, 1 = benign
- Problem type: Binary classification

The dataset satisfies the minimum assignment requirement of at least 12 features and 500 instances.

## c. GitHub Repository Link
[PASTE_YOUR_GITHUB_REPOSITORY_LINK_HERE](https://github.com/Ritakaushik22/ML-assignment2)

## Live Streamlit App Link
[PASTE_YOUR_STREAMLIT_APP_LINK_HERE](https://ml-assignment2-jdxgfpwp6c6y4ilu4cuqpb.streamlit.app)

## d. Models Used and Comparison Table
The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier, Ensemble Model
6. Support Vector Machine, additional model included because the assignment text mentions 6 models

### Model Evaluation Metrics
| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree            |     0.9211 | 0.9163 |      0.9565 |   0.9167 | 0.9362 | 0.8341 |
| kNN                      |     0.9561 | 0.9788 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Naive Bayes              |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) |     0.9561 | 0.9931 |      0.9589 |   0.9722 | 0.9655 | 0.9054 |
| Support Vector Machine   |     0.9825 | 0.995  |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

## Observations on Model Performance
| ML Model Name                    | Observation about model performance                                                                                                                    |
|:---------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression              | Performed strongly because the numeric diagnostic features separate well after scaling. It gives strong AUC and balanced precision-recall performance. |
| Decision Tree                    | Easy to interpret and useful for decision rules, but it may overfit compared with ensemble methods. Performance depends on tree depth.                 |
| kNN                              | Performed well after feature scaling because it relies on distance between samples. It can be sensitive to noisy features and the value of k.          |
| Naive Bayes                      | Fast and simple. It assumes feature independence, so performance may reduce when medical measurement features are correlated.                          |
| Random Forest (Ensemble)         | Usually strong on tabular numeric datasets because it combines many trees and reduces overfitting.                                                     |
| Support Vector Machine           | Strong after scaling. The RBF kernel can capture non-linear boundaries and often gives high AUC and F1.                                                |
| Overall Winner for your dataset? | Logistic Regression is the overall winner based on the highest F1 score. Also consider AUC and MCC for final interpretation.                           |

## Project Structure
```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- random_forest_ensemble.pkl
    |-- support_vector_machine.pkl
    |-- model_metrics.csv
    |-- feature_names.json
```

## How to Run Locally
1. Clone the GitHub repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run Streamlit app:
```bash
streamlit run app.py
```
4. Upload `test_data.csv` or use the default data included in the repository.
5. Select a model from the dropdown to view metrics, confusion matrix, classification report, and predictions.

## Streamlit App Features
- CSV upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix
- Classification report
- Prediction table

## Deployment Steps Followed
1. Go to https://streamlit.io/cloud
2. Sign in using GitHub account
3. Click New App
4. Select repository
5. Choose branch, usually main
6. Select `app.py`
7. Click Deploy

## Final Submission Checklist
- GitHub repository link works
- Streamlit app link opens correctly
- App loads without errors
- All required features are implemented
- README.md content is added to submitted PDF
- Screenshot from BITS Virtual Lab is added to submitted PDF
