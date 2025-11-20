from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
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

# Diabetes (Outcome)
model = BayesianNetwork([
    ('Age', 'Pregnancies'), ('Age', 'Glucose'), ('Age', 'BloodPressure'), ('Age', 'Diabetes'), 
    ('DiabetesPedigreeFunction', 'Diabetes'), ('DiabetesPedigreeFunction', 'BMI'), ('DiabetesPedigreeFunction', 'Insulin'),
    ('BMI', 'SkinThickness'), ('BMI', 'BloodPressure'), ('BMI', 'Glucose'), ('BMI', 'Insulin'), ('BMI', 'Diabetes'),
    ('Glucose', 'Insulin'), ('Glucose', 'Diabetes'),
    ('Pregnancies', 'Diabetes')
])

