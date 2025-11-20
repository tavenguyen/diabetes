from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Diabetes (Outcome)
model = BayesianNetwork([
    ('Age', 'Pregnancies'), ('Age', 'Glucose'), ('Age', 'BloodPressure'), ('Age', 'Diabetes'), 
    ('DiabetesPedigreeFunction', 'Diabetes'), ('DiabetesPedigreeFunction', 'BMI'), ('DiabetesPedigreeFunction', 'Insulin'),
    ('BMI', 'SkinThickness'), ('BMI', 'BloodPressure'), ('BMI', 'Glucose'), ('BMI', 'Insulin'), ('BMI', 'Diabetes'),
    ('Glucose', 'Insulin'), ('Glucose', 'Diabetes'),
    ('Pregnancies', 'Diabetes')
])

print(model.nodes())
print(model.edges())

