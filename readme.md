# Data Mining Course Project – Project Ⅳ

## 1. Project Introduction

In this project, we selected **Project Ⅳ: Data Mining Practice on Real World Task – Predict Diabetes at Early Stage**.  
The goal of this project is to build a machine learning classifier to predict whether a patient has diabetes based on medical symptoms and demographic information.

The dataset contains patient records including age, gender, and various symptoms related to diabetes.  
We performed data preprocessing and trained several classification models to complete the prediction task.

Specifically, we implemented and compared three machine learning algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

After training, the models are saved locally and can be directly used for testing on the dataset.

---

## 2. Project Structure
data_mining/
│
├── data/
│ └── data.csv
│
├── model/
│ ├── logistic_regression.pkl
│ ├── svm.pkl
│ ├── random_forest.pkl
│ └── scaler.pkl
│
├── train.py
└── run.py


**Description**

- `data/data.csv` – Dataset used for training and testing  
- `model/` – Directory containing trained classifiers  
- `train.py` – Script used to train models and save them  
- `run.py` – Script used to load trained models and evaluate them on the test set  

---

## 3. How to Run

To check the experiment results directly using the trained models, run:

- `python run.py`


The script will:

1. Load the dataset  
2. Apply the same preprocessing steps  
3. Load trained models from the `model` directory  
4. Evaluate the models on the test set  
5. Print the evaluation results