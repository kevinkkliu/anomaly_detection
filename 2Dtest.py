import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import tensorflow as tf
import math

data = pd.read_csv('',usecols=(1,2),nrows=30000)
label = pd.read_csv("train.csv",usecols=['fraud_ind'],nrows= 30000)
label = label.iloc[:,0]
x=data.iloc[:,0]
y=data.iloc[:,1]
t= np.arctan2(y,x)

plt.scatter(x,y,marker=',', c=label,s=0.5)

plt.savefig('2dtest',dpi=1500)
