# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the Dependent variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score :", (accuracy_score(y_test, y_pred))*100, "%")

# Predicting a new result

age = int(input("Enter Age of patient : "))
gender = int(input("Enter Gender of patient [1 for Male | 0 for Female] : "))
heartRate = float(input("Enter Heart Rate of patient [in BPM (Beats Per Minute)] : "))
sbp = float(input("Enter Systolic Blood Pressure of patient : "))
dbp = float(input("Enter Diastolic Blood Pressure of patient : "))
bloodSugar = float(input("Enter Blood Sugar of patient : "))
ckmb = float(input("Enter Creatine Kinase-Myoglobin Binding [CK-MB] of patient [in ng/mL] : "))
troponin = float(input("Enter Level of Troponin of patient [in ng/mL] : "))
msg1 = "| Very High Chance of Heart Attack |"
msg2 = "| Low Chance of Heart Attack |"
prediction = classifier.predict(sc.transform([[age, gender, heartRate, sbp, dbp, bloodSugar, ckmb, troponin]]))
if prediction == 1:
  print("\n")
  print("-"*len(msg1))
  print(msg1)
  print("-"*len(msg1))
elif prediction == 0:
  print("\n")
  print("-"*len(msg2))
  print(msg2)
  print("-"*len(msg2))