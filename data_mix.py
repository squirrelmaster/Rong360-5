import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import gc
import os
import time

time1 = time.clock()
if not os.path.exists('data_mix'):
    os.mkdir('data_mix')
file_1 = pd.read_csv("open_data_train_valid/train/train_1.txt",chunksize=100,sep='\t')
file_1 = file_1.get_chunk()
columns = file_1.columns
data_type = file_1.dtypes.copy()
id_1 = data_type[data_type== 'int64'].index
id_2 = data_type[data_type== 'float64'].index
d = {}
for each in id_1:
    d[each] = np.int16
for each in id_2:
    d[each] = np.float16
def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return '{:03.2f}MB'.format(usage_mb)

time2 = time.clock()
print('stage 1: %f used' %(time2-time1))
names = []
for i in range(2,6):
    names.append('open_data_train_valid/train/train_%d.txt' %i)
labelled_iter = pd.read_csv("open_data_train_valid/train/train_1.txt",chunksize=20000,sep='\t',dtype =d)
total = labelled_iter.get_chunk()

time3 = time.clock()
print('stage 2: %f' %(time3-time2))
for name in names:
    unlabelled_iter = pd.read_csv(name,chunksize=20000,sep='\t',dtype =d,names = columns)
    unlabelled = unlabelled_iter.get_chunk()
    total = pd.concat([total,unlabelled],axis=0)
    del unlabelled
    gc.collect()

time4 = time.clock()
print('stage 3: %f' %(time4-time3))
unlabeled_index = total['label'][total['label'].isnull() == True].index
labeled_index = total['label'][total['label'].isnull() == False].index
tag = total.pop('tag')
label = total.pop('label')
load_dt = total.pop('loan_dt')
id = total.pop('id')
missing_value = (total.isnull().sum()/total.isnull().count()).sort_values(ascending = False)
total.drop(missing_value[missing_value>0.8].index,1,inplace = True)
Q3 = total.quantile(q = 0.75,axis = 0)
Q1 = total.quantile(q = 0.25,axis = 0)
Q_up = Q3+1.5*(Q3-Q1)
Q_low = Q3-1.5*(Q3-Q1)
scaled = total.ge(Q_low.T,axis=1)&total.le(Q_up.T,axis=1)
total = total[scaled]

time5 = time.clock()
print('stage 4: %f s' %(time5-time4))
del Q1
del Q3
del Q_low
del Q_up
del file_1
del id_1
del id_2
categorical_indices = total.dtypes[total.dtypes == 'int16'].index
numerical_indcies = total.columns.drop(categorical_indices)
for each in total[numerical_indcies].columns:
    total[numerical_indcies][each].fillna(total[numerical_indcies][each].median(),inplace = True)
total[categorical_indices].fillna(-5,inplace = True)
for each in total[numerical_indcies].columns:
    total[each].fillna(total[each].median(),inplace = True)
'''
for each in total[numerical_indcies].columns:
    total[each]= total[each].apply(lambda x: (x-x.mean())/x.std())
'''
pd.DataFrame.to_csv(total,'data_mix/new_data.txt',sep = '\t')
#y_pred = KMeans(n_clusters = 4,max_iter= 1000).fit_predict(total)