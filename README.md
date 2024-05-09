# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3.Train your model -Fit model to training data -Calculate mean salary value for each subset
4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5.Tune hyperparameters -Experiment with different hyperparameters to improve performance
6.Deploy your model Use model to make predictions on new data in real-world application.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KAVI NILAVAN DK
RegisterNumber:  212223230103
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:
#### Initial dataset:
![ML 7 1](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/c060cea2-b113-4606-b2c8-60dc38fbf8eb)
#### Data Info:
![ML 72](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/3c2668d3-8ae6-4b9c-82c8-c424a1a0747b)
#### Optimization of null values:
![ML 73](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/06654521-c605-420a-8b84-d575dbb2c7b5)
#### Converting string literals to numericl values using label encoder:
![ML 74](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/e0acc5f9-6ae6-416c-b383-e3a2dc3b267d)
#### Assigning x and y values:
![ML75](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/79e4f455-8c69-4465-a607-7305bb4585f0)
#### Mean Squared Error:
![ML 76](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/4e536a29-98a3-42b3-8b7f-3b537dbc1543)
#### R2 (variance):
![ML 77](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/d8f30db4-0b8a-42d3-a651-84eba5c1c786)
#### Prediction:
![ML 78](https://github.com/KavinilavanDK/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144870429/7b80255e-2fec-41b9-aad3-f5bba582300e)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
