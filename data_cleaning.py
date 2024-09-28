import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
data = pd.read_csv(r'c:/Users/kawthar/Downloads/train.csv')
x=data.iloc[:,[2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y=data.iloc[:,-1].values
y=y.reshape(-1,1)

imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
x=imputer.fit_transform(x)
y=imputer.fit_transform(y)
print(x)
print(y)