<h1 align="center" id="title">Diabetes Probability</h1>

<p align="center"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="shields"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="shields"></p>

<p id="description">This is very simple model using Bayesian network to predict Diabetes based on (Pregnancies, Glucose, Blood Pressure,Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome).</p>

| **Node**           	| **Threshold**   	| **Meaning**                                     	|
|--------------------	|-----------------	|-------------------------------------------------	|
| _Glucose_          	| 140 mg/dL       	| 0: Glucose < 140 mg/dL 1: Glucose >= 140 mg/dL  	|
| _BMI_              	| 30 kg/m^2       	| 0: < 30, 1 > 30                                  	|
| _BloodPressure_    	| 80 mm Hg        	| 0: < 80, 1: >= 80                                	|
| _Age_              	| 30              	| 0: < 30, 1: >= 30                               	|
| _Pregnancies_      	| 3 or 4          	| 0: < 3, 1: >= 3                                 	|
| _Insulin_          	| 100-120 mu U/ml 	| 0: < 120, 1: >= 120                             	|
| _SkinThickness_    	| 27 - 29mm       	| 0: < 29, 1: >= 20                               	|
| _PedigreeFunction_ 	| 0.47            	| 0: < 0.47, 1: >= 0.47                           	|

## 📐 Bayesian Estimation Formula

To compute the conditional probability:

```math
P(C = k \mid \text{Parents} = j)
=
\frac{N_{\text{obs}}(C = k, \text{Pa} = j) + \alpha(C = k, \text{Pa} = j)}
{N_{\text{total\_obs}}(\text{Pa} = j) + \alpha_{\text{total}}(\text{Pa} = j)}
```

- $N_{\text{obs}}(C = k, \text{Pa}=j)\$: Number of real occurrences in the dataset where child \(C\) takes value \(k\) and its parents match configuration \(j\).
- $\alpha(C = k, \text{Pa}=j)$: Prior belief about how often each combination occurs (also called *pseudo counts*).
- $N_{\text{total obs}}$: Total number of observed rows with parent configuration $j$.
- $\alpha_{\text{total}}(\text{Pa}=j)$: Sum of all pseudo-counts for all child values under this parent configuration.

## 📊 Example Dataset

```python
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
data = pd.DataFrame(
    data={
        "A": [0, 0, 1],
        "B": [0, 1, 0],
        "C": [1, 1, 0]
    }
)

model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
estimator = BayesianEstimator(model, data)

cpd_C = estimator.estimate_cpd(
    node="C",
    prior_type="dirichlet",
    pseudo_counts=[[1, 1, 1, 1], [2, 2, 2, 2]],
)

print(cpd_C)
```


### Pseudo-count Table

| **C value** 	| **(A = 0, B = 0)** 	| **(A = 0, B = 1)** 	| **(A = 1, B = 0)** 	| **(A = 1, B = 1)** 	|
|-------------	|--------------------	|--------------------	|--------------------	|--------------------	|
| 0           	| 1                  	| 1                  	| 1                  	| 1                  	|
| 1           	| 2                  	| 2                  	| 2                  	| 2                  	|

### CPD Result
|   **A**  	| **A(0)** 	| **A(0)** 	| **A(1)** 	|  **A(1)** 	|
|:--------:	|:--------:	|:--------:	|:--------:	|:---------:	|
|   **B**  	|   B(0)   	|   B(1)   	|   B(0)   	|    B(1)   	|
| **C(0)** 	|   0.25   	|   0.25   	|    0.5   	| 0.3333333 	|
| **C(1)** 	|   0.75   	|   0.75   	|    0.5   	| 0.6666666 	|

<h2>🛠️ Installation Steps:</h2>

<p>1. Conda Environment</p>

```
conda env create -f environment.yml
```

<p>2. Clone Repository</p>

```
git clone https://github.com/tavenguyen/diabetes.git
```

### References
- pgmpy Documentation 
- Bayesian Estimator 
  
 
