# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Logesh.N.A
RegisterNumber: 212223240078
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
  
*/
```

## Output:

## DATASET:
![IMG-20240229-WA0022](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/44f6f1e8-48f1-4430-8705-211c1d3dc21a)
## HEAD VALUES:
![IMG-20240229-WA0020](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/9336bee5-738f-4d07-9472-e4f88b9d5b2a)

## TAIL VALUES:
![IMG-20240229-WA0018](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/b9dbc00a-4302-4dcb-a1c6-94b1fa31fbf1)

## X and Y VALUES:
![IMG-20240229-WA0016](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/64654c82-da94-4b73-a1ff-80ba5de4b980)

## Predication values of X and Y:
![IMG-20240229-WA0023](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/9fffef1e-34aa-49b0-b635-643c61731080)

## MSE,MAE and RMSE:
![IMG-20240229-WA0021](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/7b51a3ad-5fac-4f1f-8c41-5822e0f008a8)

## Training Set:
![IMG-20240229-WA0024](https://github.com/Logesh051/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979188/0928cb01-6434-4a81-8c3b-013ab1d140c9)

## Testing Set:
![Uploading IMG-20240229-WA0019.jpg…]()

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
