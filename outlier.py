import numpy as np
from matplotlib import pyplot as plt

# Original data
data = [10, 386, 479, 627, 20, 523, 482, 483, 542, 699, 535, 617, 577, 471, 615, 583, 441, 562, 543]
print(data)
# Convert the data to a numpy array
data = np.array(data)
print(data)
# Calculate the mean of the data
mean = round(np.mean(data),2)
# Calculate the std of the data
std = round(np.std(data),2)
threshold=3
# Calculate the upper and lower bounds
lower_bound = mean - threshold * std
upper_bound = mean + threshold * std
print(mean)
print(std)
print(lower_bound)
print(upper_bound)
plt.title("Histogram with outlier")
plt.hist(data, bins=[0, 100, 200, 300, 400, 500, 600, 700, 800], alpha=0.7, color='g')
plt.show()
print("OUTLIER")
for x in data:
    if x < lower_bound or x > upper_bound:
        print(x)
        print("END OUTLIER")

print("Cleaning data")
# Create a dataset without outliers
final_list = [x for x in data if (x > lower_bound) and (x < upper_bound)]
filtered_elements = np.array(final_list)

# Print the cleaned data and outliers
print("Cleaned Data:", filtered_elements)

plt.title("Histogram without outlier")
plt.hist(filtered_elements, bins = [0, 100, 200, 300, 400, 500, 600, 700, 800], alpha=0.7, color='g')
plt.show()