#importing labraries
import numpy as np 
import pandas as pd
#importing dataset
data = pd.read_csv(r'c:/Users/kawthar/Downloads/train.csv')

print(data.columns)
print(data.head())
print(data.isnull().sum())
print(data.describe)
print(data[['Evaporation','Sunshine']])
x=data.iloc[:,[2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y=data.iloc[:,-1].values
y=y.reshape(-1,1)
print(x)
print(y)
