import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


MODEL_DIR = "./model"


# ==============================
# 1 读取数据
# ==============================

data = pd.read_csv("./data/data.csv")

print("数据集大小:", data.shape)


# ==============================
# 2 数据预处理
# ==============================

mapping = {
    "Male": 1,
    "Female": 0,
    "Yes": 1,
    "No": 0,
    "Positive": 1,
    "Negative": 0
}

data = data.replace(mapping)

X = data.drop("class", axis=1)
y = data["class"]


# ==============================
# 3 数据划分（必须和训练一致）
# ==============================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("测试集大小:", X_test.shape)


# ==============================
# 4 加载 scaler
# ==============================

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

X_test = scaler.transform(X_test)


# ==============================
# 5 模型列表
# ==============================

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm.pkl",
    "Random Forest": "random_forest.pkl"
}


# ==============================
# 6 测试模型
# ==============================

print("\n========== 测试结果 ==========")

for name, file in model_files.items():

    model_path = os.path.join(MODEL_DIR, file)

    model = joblib.load(model_path)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print("\n==============================")
    print("模型:", name)
    print("==============================")

    print("Accuracy :", round(acc,4))
    print("Precision:", round(prec,4))
    print("Recall   :", round(rec,4))
    print("F1-score :", round(f1,4))
    print("ROC-AUC  :", round(auc,4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred))