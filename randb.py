
import pandas as pd
import numpy as np
from sklearn import preprocessing as p
#Creating a DataFrame
data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
df2 = pd.DataFrame(data)
print("\n ORIGINAL DATA VALUES")
print("------------------------")
print(df2)

#Scaling the data
scaler = p.MinMaxScaler(feature_range=(0,5))
data=scaler.fit_transform(df2)
print(data.round(2))
#binarizing
scaler = p.Binarizer(threshold=5).transform(df2)
print(scaler.round(2))