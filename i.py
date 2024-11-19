#Program 1: Filling Missing Numerical Values in a DataFrame
#• Task: Use Python code to create a DataFrame with missing numerical values and fill
#them using two standard techniques.
#• Feedback: Techniques that can be used - e.g., mean or median imputation, droppingrowswith missing values, or filling missing values with 0’s)
import pandas as pd
import numpy as np
#data={
 #   'a':[1,2,3,4,5,np.nan],
  #  'b':[45,6,332,2323,np.nan,np.nan]
#}
data1=np.random.rand(5,3)
ind=['a','c','e','f','h']
col=['C1','C2','C3']
df=pd.DataFrame(data1,index=ind,columns=col)
print(df)
print("redindexing")
d1=df.reindex(['a','b','c','d','e','f','g','h'])
print(d1)
#drop missing value
print("Dropping missing value");
d=d1.dropna()
print(d)
print("filling missing value with zero");
d=d1.fillna(0)
print(d)
print("filling missing value with median");
d=d1.fillna(df.median())
print(d)