from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

import numpy as np 
import pandas as pd

# Loading data
data = pd.read_csv(r'train.csv')
testdata = pd.read_csv(r'test.csv')

print(data.columns)
num_columns = data.shape[1]
print(f'Number of columns: {num_columns}')

x = data.iloc[:, [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]].values
y = data.iloc[:, -1].values
y = y.ravel()

x_test = testdata.iloc[:, [2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]].values
y_test = testdata.iloc[:, -1].values
y_test = y_test.ravel()

print(x_test)

# Data cleaning
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x = imputer.fit_transform(x)
x_test = imputer.transform(x_test)

# Encoding data
le1 = LabelEncoder()
x[:, 0] = le1.fit_transform(x[:, 0])
x_test[:, 0] = le1.transform(x_test[:, 0])

le2 = LabelEncoder()
x[:, 4] = le2.fit_transform(x[:, 4])
x_test[:, 4] = le2.transform(x_test[:, 4])

le3 = LabelEncoder()
x[:, 6] = le3.fit_transform(x[:, 6])
x_test[:, 6] = le3.transform(x_test[:, 6])

le4 = LabelEncoder()
x[:, 7] = le4.fit_transform(x[:, 7])
x_test[:, 7] = le4.transform(x_test[:, 7])

le5 = LabelEncoder()
x[:, 18] = le5.fit_transform(x[:, 18])
x_test[:, 18] = le5.transform(x_test[:, 18])

le6 = LabelEncoder()
y = le6.fit_transform(y)
y_test = le6.transform(y_test)
print(x)
print(y)

# Feature scaling
sc = StandardScaler()
x = sc.fit_transform(x)
x_test = sc.transform(x_test)

print(x)
print(x_test)

# Balancing data
print('before',pd.Series(y).value_counts())
print('before',pd.Series(y_test).value_counts())
smote = SMOTE(random_state=0)
x_resampled, y_resampled = smote.fit_resample(x, y)
print('after', pd.Series(y_resampled).value_counts()) 

# Training Model
classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0, class_weight='balanced')
classifier.fit(x_resampled, y_resampled)
score = classifier.score(x_resampled, y_resampled)

print(score)

y_pred = classifier.predict(x_test)

f1 = f1_score(y_test, y_pred, pos_label=1)
print(f1)




print(y_pred) 
print(y_test)  

y_pred_labels = le6.inverse_transform(y_pred)
y_test_labels = le6.inverse_transform(y_test)

df = np.concatenate((y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
dataframe = pd.DataFrame(df, columns=['rain tomorrow (encoded)', 'prediction (encoded)'])

print(dataframe.head())
print(dataframe) 

submission = pd.DataFrame({
    'day': np.arange(len(y_pred_labels)),
    'rain tomorrow': y_pred_labels  
})


submission.to_csv('submission.csv', index=False)
print(submission.head())



