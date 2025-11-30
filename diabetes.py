from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from typing import Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Medical Standards Threshold (Dựa trên Y khoa)
GLUCOSE_THRESHOLD = 140 # (>= 140 : 1 | < 140: 0) 
BMI_THRESHOLD = 30
BLOOD_PRESSURE_THRESHOLD = 80
AGE_THRESHOLD = 30 # 30 or 50 

# Statistical/Research Threshold (Dựa trên thống kê)
PREGNANCIES_THRESHOLD = 3
INSULIN_THRESHOLD = 100
SKIN_THICKNESS_THRESHOLD = 29
PEDIGREE_FUNCTION_THRESHOLD = 0.47

def getThreshold(key: Any):
    match key:
        case 'Glucose':
            return GLUCOSE_THRESHOLD
        case 'BMI':
            return BMI_THRESHOLD
        case 'BloodPressure':
            return BLOOD_PRESSURE_THRESHOLD
        case 'Age':
            return AGE_THRESHOLD
        case 'Pregnancies':
            return PREGNANCIES_THRESHOLD
        case 'Insulin':
            return INSULIN_THRESHOLD
        case 'SkinThickness':
            return SKIN_THICKNESS_THRESHOLD
        case 'DiabetesPedigreeFunction':
            return PEDIGREE_FUNCTION_THRESHOLD
        case _:
            return 0.0

def process(evidence : dict):
    processsed_evidence = {}
    for key in evidence:
        processsed_evidence[key] = int(evidence[key] >= getThreshold(key))
    return processsed_evidence

# Read CSV
df = pd.read_csv("diabetes_dataset.csv")
total_samples = len(df)

cols_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
means = dict()
for key in cols_missing:
    for value in df[key]:
        means[key] = means.get(key, 0.0) + value
    means[key] = means[key] / total_samples

for key in cols_missing:
    df[key] = df[key].replace(0, means[key])

y = df['Outcome']

# Tham số quan trọng: stratify=y
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước Train: {len(X_train)}")
print(f"Kích thước Test:  {len(X_test)}")

# Kiểm tra xem tỷ lệ bệnh nhân có được bảo toàn không?
print("\nTỷ lệ người bệnh (Outcome=1):")
print(f"- Gốc:   {df['Outcome'].mean():.2%}")
print(f"- Train: {X_train['Outcome'].mean():.2%}")
print(f"- Test:  {X_test['Outcome'].mean():.2%}")

# Nodes
df['Glucose'] = (df['Glucose'] >= GLUCOSE_THRESHOLD).astype(int)
df['BMI'] = (df['BMI'] >= BMI_THRESHOLD).astype(int)
df['BloodPressure'] = (df['BloodPressure'] >= BLOOD_PRESSURE_THRESHOLD).astype(int)
df['Age'] = (df['Age'] >= AGE_THRESHOLD).astype(int)
df['Pregnancies'] = (df['Pregnancies'] >= PREGNANCIES_THRESHOLD).astype(int)
df['Insulin'] = (df['Insulin'] >= INSULIN_THRESHOLD).astype(int)
df['SkinThickness'] = (df['SkinThickness'] >= SKIN_THICKNESS_THRESHOLD).astype(int)
df['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] >= PEDIGREE_FUNCTION_THRESHOLD).astype(int)

y = df['Outcome']

# Tham số quan trọng: stratify=y
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

print("\n(Outcome=1):")
print(f"Train: {X_train['Outcome'].mean():.2%}")
print(f"Test:  {X_test['Outcome'].mean():.2%}")

# Diabetes (Outcome)
model = BayesianNetwork([
    ('Age', 'Pregnancies'), ('Age', 'Glucose'), ('Age', 'BloodPressure'), ('Age', 'Outcome'), 
    ('DiabetesPedigreeFunction', 'Outcome'), ('DiabetesPedigreeFunction', 'BMI'), ('DiabetesPedigreeFunction', 'Insulin'),
    ('BMI', 'SkinThickness'), ('BMI', 'BloodPressure'), ('BMI', 'Glucose'), ('BMI', 'Insulin'), ('BMI', 'Outcome'),
    ('Glucose', 'Insulin'), ('Glucose', 'Outcome'),
    ('Pregnancies', 'Outcome')
])

model.fit(X_train, estimator=MaximumLikelihoodEstimator)
model.check_model()

test_data_no_label = X_test.drop(columns=['Outcome'])
y_pred_df = model.predict(test_data_no_label)
y_pred = y_pred_df['Outcome']
y_true = X_test['Outcome']

acc = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"(Accuracy): {acc:.4f}")
print("\n(Confusion Matrix):")
print(conf_matrix)
print(report)