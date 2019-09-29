import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA

## Get working directory
PATH = os.getcwd()

## Load data
df = pd.read_csv("test.csv",usecols=['acqic','conam','contp','csmcu','ecfg','etymd','flbmk','flg_3dsmk','hcefg','insfg','iterm','locdt','loctm','mcc','mchno','ovrlt','scity','stocn','stscd','txkey'])

#提出label
label = pd.read_csv("train.csv",usecols=['fraud_ind'])

#filter Y&N
data = df.applymap(lambda m: 1 if m == 'Y' else(0 if (m =='N'or math.isnan(m)) else m))

#normalization
data =preprocessing.scale(data)

# Generating PCA and
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(data))
#df_pca = df_pca.values

## TensorFlow Variable from data
tf_data = tf.Variable(df_pca)

print(df_pca)
df_pca.to_csv('3DtestData.csv')

