from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
import datetime
from pandas import Series
from collections import Counter
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV,cross_val_predict   #Perforing grid search
import sys,random
from sklearn.cross_validation import StratifiedKFold
PATH = '../../media/oem/Data_disk_1/weiwu/jupyter_notebook/users/xiayang/'

def get_data():
    file_1 = pd.read_csv(path + 'train_1.txt', chunksize=100, sep='\t')
    file_1 = file_1.get_chunk()
    columns = file_1.columns
    # add_feature = np.append(new_feature,'label')
    # train data
    print('load training data')

    table_1 = pd.read_table(path + 'train_1.txt', sep='\t', chunksize=20000)
    table_2 = pd.read_table(path + 'train_2.txt', sep='\t', chunksize=20000, names=columns)
    table_1 = table_1.get_chunk()
    table_2 = table_2.get_chunk()
    data = pd.concat([table_1, table_2], axis=0, ignore_index=True)
    label = data['label']
    label = label.iloc[label[label.notnull()].index.values]
    data = data.iloc[label[label.notnull()].index.values]
    data.drop(['label'], axis=1, inplace=True)
    data.drop(['loan_dt', 'tag', 'id'], axis=1, inplace=True)

    del table_1
    del table_2

    return data,label

def write_file(filename, proba):
    with open(filename, 'w') as f:
        f.write('id,prob')
        f.write('\n')
        for i in range(proba.shape[0]):
            line = str(100000 + i + 1) + ',' + str(proba[i])
            f.write(line + '\n')

def xgb_model(seed=1,gamma=0.1,max_depth=12,lamb=300,subsample=0.7,colsample_bytree=0.5,min_child_weight=1,scale_pos_weight=1):
    model=XGBClassifier(n_estimators=350 ,nthread=8,  learning_rate=0.08, gamma=gamma,max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample,
                        colsample_bytree=colsample_bytree,objective='binary:logistic',seed=seed,reg_lambda=lamb,scale_pos_weight=scale_pos_weight)
    return model

def xgb_param(k):
    random_seed = range(2017)
    gamma = [i/1000.0 for i in range(50,300,10)]
    max_depth = [5,6,7,8]
    lambd = range(200,400,5)
    subsample = [i/1000.0 for i in range(500,800,10)]
    colsample_bytree = [i/1000.0 for i in range(250,450,10)]
    min_child_weight = [i/1000.0 for i in range(1000,4000,30)]
    scale_pos_weight = [i for i in range(2,6)]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    random.shuffle(scale_pos_weight)

    model = xgb_model(random_seed[k],gamma[k],max_depth[k%4],lambd[k],subsample[k],colsample_bytree[k],min_child_weight[k],scale_pos_weight[k%4])
    return model


def bagged_set(X,y,estimators, X_test):
    bagged_pred = np.zeros(X_test.shape[0])
    for n in range(0, estimators):
        print('{0} model'.format(n))
        model = xgb_param(n)
        model.fit(X,y)
        preds = model.predict_proba(X_test)[:,1]

        bagged_pred += preds
    bagged_pred /= estimators

    return bagged_pred

def main():
    PATH = '../../media/oem/Data_disk_1/weiwu/jupyter_notebook/users/xiayang/'

    print('loading data')
    data,label = get_data()

    print('split training and testing data')
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=None)

    del data

    print('start training')
    mean_auc = 0.0
    bagging = 5
    n = 4
    kfold = StratifiedKFold(y_train,n_folds= n ,shuffle= True, random_state= 1)
    i = 0

    for train_index, test_index in kfold:
        #cv
        X_train, X_cv = X_train[train_index],X_train[test_index]
        y_train, y_cv = y_train[train_index],y_train[test_index]

        print('train size: %d. test size: %d, cols %d' %((X_train.shape[0]),(X_cv.shape[0]),(X_train.shape[1])))

        #train and predict
        preds = bagged_set(X_train,y_train,bagging, X_cv)

        roc_auc = roc_auc_score(y_cv,preds)
        print('AUC (fold %d/%d): %f' %(i+1,n,roc_auc))
        mean_auc += roc_auc

        i += 1
    mean_auc /= n
    print('Average AUC: %f' %(mean_auc))

    preds = bagged_set(X_train,y_train,bagging,X_test)
    roc_auc = roc_auc_score(y_test,preds)

    print('test AUC: %f ' %(mean_auc))

