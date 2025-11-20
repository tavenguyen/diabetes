from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from typing import Any
import pandas as pd
import numpy as np

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
    for key in evidence:
        evidence[key] = int(evidence[key] >= getThreshold(key))

# Read CSV
df = pd.read_csv("diabetes_dataset.csv")
total_samples = len(df)

# Nodes
df['Glucose'] = (df['Glucose'] >= GLUCOSE_THRESHOLD).astype(int)
df['BMI'] = (df['BMI'] >= BMI_THRESHOLD).astype(int)
df['BloodPressure'] = (df['BloodPressure'] >= BLOOD_PRESSURE_THRESHOLD).astype(int)
df['Age'] = (df['Age'] >= AGE_THRESHOLD).astype(int)
df['Pregnancies'] = (df['Pregnancies'] >= PREGNANCIES_THRESHOLD).astype(int)
df['Insulin'] = (df['Insulin'] >= INSULIN_THRESHOLD).astype(int)
df['SkinThickness'] = (df['SkinThickness'] >= SKIN_THICKNESS_THRESHOLD).astype(int)
df['DiabetesPedigreeFunction'] = (df['DiabetesPedigreeFunction'] >= PEDIGREE_FUNCTION_THRESHOLD).astype(int)

# Diabetes (Outcome)
model = BayesianNetwork([
    ('Age', 'Pregnancies'), ('Age', 'Glucose'), ('Age', 'BloodPressure'), ('Age', 'Outcome'), 
    ('DiabetesPedigreeFunction', 'Outcome'), ('DiabetesPedigreeFunction', 'BMI'), ('DiabetesPedigreeFunction', 'Insulin'),
    ('BMI', 'SkinThickness'), ('BMI', 'BloodPressure'), ('BMI', 'Glucose'), ('BMI', 'Insulin'), ('BMI', 'Outcome'),
    ('Glucose', 'Insulin'), ('Glucose', 'Outcome'),
    ('Pregnancies', 'Outcome')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)
model.check_model()

# Result
infer = VariableElimination(model)

# P(Outcome | Age = 50, Glucose = 130, BMI = 29)
dist_evidence = {
    'Age': 50,
    'Glucose': 130,
    'BMI': 29
}
process(dist_evidence)
w_dist = infer.query(['Outcome'], dist_evidence)
print(w_dist)