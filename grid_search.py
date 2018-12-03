from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import datetime
from pandas import Series
from collections import Counter

new_feature_csv = pd.read_csv('valid_data/2800 features.csv')
new_feature = new_feature_csv['feature'].values

file_1 = pd.read_csv('open_data_train_valid/train/train_1.txt',chunksize=100,sep='\t')
file_1 = file_1.get_chunk()
columns = file_1.columns
add_feature = np.append(new_feature,'label')
#train data
print('load training data')


table_1 = pd.read_table('open_data_train_valid/train/train_1.txt', sep='\t', chunksize=20000,usecols = add_feature)
table_2 = pd.read_table('open_data_train_valid/train/train_2.txt', sep='\t', chunksize=20000, names=columns,usecols = add_feature)
table_1 = table_1.get_chunk()
table_2 = table_2.get_chunk()
data = pd.concat([table_1, table_2], axis=0, ignore_index=True)
label = data['label']
label = label.iloc[label[label.notnull()].index.values]
data = data.iloc[label[label.notnull()].index.values]
data.drop(['label'], axis=1, inplace=True)

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV,cross_val_predict   #Perforing grid search

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 50, 24

param_test = {
  'max_depth':[ 3,4,5],
  'min_child_weight':[1,2,3]
}
gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=5,
gamma=0.05, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=2, scale_pos_weight=15,seed=27,slient = False,),
 param_grid = param_test, scoring='roc_auc',n_jobs=2,iid=False, cv=4,verbose = 10)
gsearch.fit(data,label)
print(gsearch.cv_results_,gsearch.best_params_, gsearch.best_score)c