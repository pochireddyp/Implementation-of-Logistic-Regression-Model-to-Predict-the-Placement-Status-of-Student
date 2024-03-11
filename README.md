# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the python library pandas
2.Read the dataset of Placement_Data 
3.Copy the dataset in data1 
4.Remove the columns which have null values using drop()
5.Import the LabelEncoder for preprocessing of the dataset
6.Assign x and y as status column values 
7.From sklearn library select the model to perform Logistic Regression
8.Print the accuracy, confusion matrix and classification report of the dataset
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: POCHI REDDY
RegisterNumber:  212223240115
*/
```
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

![ml-4 1](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150232043/e7240334-8f98-4904-b5b2-69c50476e432)

![ml-4 2](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150232043/5949f5a5-31e7-488a-8f34-70cbc25ec3cc)

![ml-4 3](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150232043/37089672-dbcd-47fb-9ec1-874890925dc3)

![ml-4 4](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/150232043/bc7693d9-86f1-460e-9e6d-3f394669a751)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
