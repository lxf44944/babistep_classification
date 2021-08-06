from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
import pandas as pd
import numpy as np
'''样本数量平衡处理 undersampling'''
#load Data
data=pd.read_csv('../Data/Data_afterEDA2.csv')

#drop same data
data=data.drop_duplicates(subset=['pName'],keep='first')
pName=pd.DataFrame(data['pName'])
Cat_l1=pd.DataFrame(data['Cat_l1'])

# print(pName)
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(pName, Cat_l1)
print(y_resampled.value_counts())

# df=pd.DataFrame({'pName':X_resampled.pName,'Cat_l1':y_resampled.Cat_l1})
# df.to_csv('../Data/Data_cleaned_afterEDA2.csv',index=False,encoding='UTF-8')
#
#
