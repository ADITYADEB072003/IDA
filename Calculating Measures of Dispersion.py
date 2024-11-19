import statistics as s
from scipy import stats as s1
import numpy as np
# Sample data.


data = [10, 386, 479, 627, 20, 523, 482, 483, 542, 699, 535, 617, 577, 471, 615, 583, 441, 562,
        563, 527, 453, 530, 433, 541, 585, 704, 443, 569, 430, 637, 331, 511, 552, 496, 484, 566, 554,
        472, 335, 440, 579, 341, 545, 615, 548, 604, 439, 556, 442, 461, 624, 611, 444, 578, 405, 487,
        490, 496, 398, 512, 422, 455, 449, 432, 607, 679, 434, 597, 639, 565, 415, 486, 668, 414, 665,
        763, 557, 304, 404, 454, 689, 610, 483, 441, 657, 590, 492, 476, 437, 483, 12, 363, 711, 543]

data1=np.array(data)
print(data1)
max_range=np.max(data)
min_range=np.min(data)
range=max_range-min_range
print("Range of the data is: ",range)
mean=round(s.stdev(data),2)
print("Standard Deviation of the data is: ",mean)
variance=s.variance(data1)
print("Variance of the data is: ",variance)
print("Aritmetic mean")
print("Mean",s1.tmean(data))
print("Harmonic mean")
print(" H Meam",s1.hmean(data))
print("Geometric mean")
print("Gmean",s1.gmean(data))
print("Median")
print("Median",s.median(data))
print("Mode")
print("Mode",s1.mode(data))
print("Median Absolute Deviation")
q1=np.percentile(data,25)
q2=np.percentile(data,50)
q3=np.percentile(data,75)
iqr=q3-q1
print("IQR",iqr)
print("Semi-IQR",iqr/2)
print("Q2",q2)