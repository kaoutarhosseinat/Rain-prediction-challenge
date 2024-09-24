import pandas as pd
data = pd.read_csv('c:/Users/kawthar/Downloads/train.csv')
print(data.head())
print(data.isnull().sum())
print(data.describe)
