import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ==============================
# 1 创建模型目录
# ==============================

MODEL_DIR = "./model"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# ==============================
# 2 读取数据
# ==============================

data = pd.read_csv("./data/data.csv")

print("数据集大小:", data.shape)


# ==============================
# 3 数据预处理
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
# 4 划分数据集
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

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# ==============================
# 5 标准化
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 保存 scaler
joblib.dump(scaler, MODEL_DIR + "/scaler.pkl")


# ==============================
# 6 定义模型
# ==============================

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ==============================
# 7 训练模型
# ==============================

print("\n===== 训练模型 =====")

for name, model in models.items():

    model.fit(X_train, y_train)

    # 保存模型
    model_path = f"{MODEL_DIR}/{name}.pkl"
    joblib.dump(model, model_path)

    print(f"{name} 已保存到 {model_path}")


# ==============================
# 8 验证集评估
# ==============================

print("\n===== 验证集结果 =====")

for name, model in models.items():

    pred = model.predict(X_val)
    prob = model.predict_proba(X_val)[:,1]

    acc = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred)
    rec = recall_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    auc = roc_auc_score(y_val, prob)

    print("\n模型:", name)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)


# ==============================
# 9 测试集最终评估
# ==============================

print("\n===== 测试集最终结果 =====")

for name in models.keys():

    model_path = f"{MODEL_DIR}/{name}.pkl"

    # 重新加载模型
    model = joblib.load(model_path)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    print("\n模型:", name)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)

    print("\n分类报告:")
    print(classification_report(y_test, pred))


# ==============================
# 10 特征重要性（Random Forest）
# ==============================

rf = joblib.load(MODEL_DIR + "/random_forest.pkl")

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n===== 特征重要性 =====")
print(feature_importance)