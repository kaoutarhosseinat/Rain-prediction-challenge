from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
data = pd.read_csv(r'c:/Users/kawthar/Downloads/train.csv')
print(data.columns)
num_columns = data.shape[1]
print(f'Number of columns: {num_columns}')
x=data.iloc[:,[2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y=data.iloc[:,-1].values
y=y.reshape(-1,1)
y=data.iloc[:,-1].values
#encoding data
le1 = LabelEncoder()
x[:,0] = le1.fit_transform(x[:,0])
le2 = LabelEncoder()
x[:,4] = le2.fit_transform(x[:,4])
le3 = LabelEncoder()
x[:,6] =le3.fit_transform(x[:,6])
le4 = LabelEncoder()
x[:,7] =le4.fit_transform(x[:,7])
le5 = LabelEncoder()
x[:,18] =le5.fit_transform(x[:,18])
le6 = LabelEncoder()
y=le6.fit_transform(y)
print(x)
print(y)
#feature scaling
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)
#Training Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,random_state=0)
classifier.fit(x,y)
score=classifier.score(x,y)
print(score)
y_pred = classifier.predict(x)
y_pred = le6.inverse_transform(y_pred)
print (y_pred)
