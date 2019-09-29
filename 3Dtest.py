import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import tensorflow as tf
import math

data = pd.read_csv('3DtestData.csv',usecols=(1,2,3),nrows= 1000)

x=data.iloc[:,0]
y=data.iloc[:,1]
z=data.iloc[:,2]

#data = np.array(data).tolist()

x = np.array(x)
y = np.array(y)
z = np.array(z)

"""x = tf.sigmoid(x)
y = tf.sigmoid(y)
z = tf.sigmoid(z)"""

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
x,y = np.meshgrid(x,y)


ax.contourf(x,y,z,zdir='z',offset=-2,cmap='rainbow')

ax.set_xlim((-15,15))
ax.set_ylim((-10,10))
ax.set_zlim((-5,15))

ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlabel('Z')
plt.savefig('3d')