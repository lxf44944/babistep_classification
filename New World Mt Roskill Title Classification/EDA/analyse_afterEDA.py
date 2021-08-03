from matplotlib import pyplot as plt
import matplotlib.font_manager
import numpy as np
import pandas as pd
import seaborn as sns

'''load data'''
data=pd.read_csv('../Data/Data_afterEDA2.csv',encoding='UTF-8')
count=data['Cat_l1'].value_counts()
# print(count)
'''show '''
font=matplotlib.font_manager.FontProperties(fname='../res/Songti.ttc')
sns.barplot(x=count.index,y=count.values,palette="RdBu_r")
plt.ylabel('商品数量',fontproperties=font,fontsize=14)
plt.title('EDA增强后各商品数量统计',fontproperties=font,fontsize=20)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
# plt.show()
plt.savefig('../Analyse/EDA2.png')
print(count.values)