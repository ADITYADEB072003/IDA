import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Creating a DataFrame
data = {
    'item': ['Perfume', 'Lipstick', 'Deodorant', 'Body_Lotion', 'Makeup_Set', 'Hair_Dryer', 'Lip_Gloss', 'Moisturiser', 'Scrubber'],
    'price': [3450, 1360, 2540, 1060, 1320, 3550, 3890, 1210, 2160]
}
df=pd.DataFrame(data)
print("DATASET")
print(df)

min_price=min(df['price'])
max_price=max(df['price'])
print("\nMIN PRICE:",min_price)
print("MAX PRICE:",max_price)
bins=np.linspace(min_price,max_price,4)
print("\nBINS:",bins)
labels=['low',"medium","high"]

df['price_index']=pd.cut(df['price'],bins,labels=labels,include_lowest=True)
print("\nPRICE INDEX:")
print(df)
#histogram
plt.hist(df['price'],bins=bins,edgecolor='black')
plt.axis([min_price,max_price,0,5])
plt.xlabel('Price')
plt.ylabel("no of items")
plt.show()