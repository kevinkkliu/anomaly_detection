import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from tfsvm import TFSVM

## Get working directory
PATH = os.getcwd()

## Load data
df = pd.read_csv("train.csv",usecols=['acqic','conam','contp','csmcu','ecfg','etymd','flbmk','flg_3dsmk','hcefg','insfg','iterm','locdt','loctm','mcc','mchno','ovrlt','scity','stocn','stscd','txkey'])

#提出label
label = pd.read_csv("train.csv",usecols=['fraud_ind'])
y=label['fraud_ind'].values
#filter Y&N
data = df.applymap(lambda m: 1 if m == 'Y' else(0 if (m =='N'or math.isnan(m)) else m))

#normalization
data =preprocessing.scale(data)

"""# Generating PCA and
pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(data))"""
#train test split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.5, random_state=3, stratify=y)

#Knn cross_validation(驗證此樣本是否為有效)
knn = KNeighborsClassifier(n_neighbors=100)
cv_scores = cross_val_score(knn, data, y, cv=10)
"""print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))"""

#tfsvm
tfSVM=TFSVM(learning_rate=0.1,C=0.01,display_step=20,training_epoch=50)
tfSVM.fit(X_train,y_train ,X_test,y_test)
