# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.
 
7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:S.Renuga 
RegisterNumber: 212222230118

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```

## Output:
 
## data.head()
![Screenshot 2023-10-12 203604](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/23d0fc8d-e27d-4af6-98ec-c35666e01bbd)

## data.info()

![Screenshot 2023-10-12 203739](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/bb6cc487-9c03-4755-80ef-7fb250c99d8f)

## isnull() and sum ()
![Screenshot 2023-10-12 203845](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/93ba790d-144e-4f18-a99f-39eabfd0f6e4)

## data value counts()

![Screenshot 2023-10-12 204256](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/b1309b87-4012-41b3-8a23-4b8e42373353)

## data.head() for salary

![Screenshot 2023-10-12 204620](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/e807f167-a8f2-4ec0-bc32-3433f9326f30)

## x.head()
![Screenshot 2023-10-12 204723](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/179f3011-72f3-4dc9-af59-caa79c881160)

## accuracy value

![Screenshot 2023-10-12 204815](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/88197d31-783e-4ada-8276-bc3f97b81675)

## data prediction

![Screenshot 2023-10-12 204923](https://github.com/RENUGASARAVANAN/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119292258/cfa7b225-ec81-4d5b-9fb8-a32b8ca74dec)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
