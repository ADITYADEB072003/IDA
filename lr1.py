import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df=pd.read_csv("salary_data.csv")
print(df)

x=df[['YearsExperience']]
y=df[['Salary']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


r2=r2_score(y_test,y_pred)
print("ACCURACY")
print(r2)

plt.title("TRAINING DATA")
plt.scatter(x_train,y_train,color='red',label="Training Data")
plt.plot(x_train,model.predict(x_train),color='blue',label="Regression Line")
plt.legend()
plt.show()
plt.title("TESTING DATA")

plt.scatter(x_test,y_test,color='red',label="Test Data")
plt.plot(x_test,model.predict(x_test),color='blue',label="Regression Line")
plt.legend()
plt.show()