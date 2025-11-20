from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
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

def calculateProbabilityOfNode(key):
    pass

# Read Dataset
df = pd.read_csv("diabetes_dataset.csv")
total_samples = len(df)

# Diabetes (Outcome)
model = BayesianNetwork([
    ('Age', 'Pregnancies'), ('Age', 'Glucose'), ('Age', 'BloodPressure'), ('Age', 'Diabetes'), 
    ('DiabetesPedigreeFunction', 'Diabetes'), ('DiabetesPedigreeFunction', 'BMI'), ('DiabetesPedigreeFunction', 'Insulin'),
    ('BMI', 'SkinThickness'), ('BMI', 'BloodPressure'), ('BMI', 'Glucose'), ('BMI', 'Insulin'), ('BMI', 'Diabetes'),
    ('Glucose', 'Insulin'), ('Glucose', 'Diabetes'),
    ('Pregnancies', 'Diabetes')
])

