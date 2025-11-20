<h1 align="center" id="title">Diabetes Probability</h1>

<p id="description">This is very simple model used Bayesian network to predict Diabetes based on (Pregnancies, Glucose, Blood Pressure,Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome).</p>

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

<h2>🛠️ Installation Steps:</h2>

<p>1. Conda Environment</p>

```
conda env create -f environment.yml
```

<p>2. Clone Repository</p>

```
git clone https://github.com/tavenguyen/diabetes.git
```

  
 