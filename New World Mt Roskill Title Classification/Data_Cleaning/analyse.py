import matplotlib.font_manager
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import *
#res


font=matplotlib.font_manager.FontProperties(fname='../res/Songti.ttc')
plt.figure(dpi=200,figsize=(8,8))



#data
'''需要绘制不同图时，修改对应的num_series数据源和表格文件中的相关图例名称即可'''
from Data_Cleaning import data_2
from Data_Cleaning import Multi_analyse



num_series=data_2.groupby('level_1')['PricePerItem'].agg(np.mean)
# sns.barplot(x=num_series.index,y=num_series.values,palette="RdBu_r")
# plt.ylabel('折扣商品中位数',fontproperties=font,fontsize=14)
# plt.title('第一分类下不同类别折扣商品中位数统计',fontproperties=font,fontsize=20)
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.3)
# # plt.show()
# plt.savefig('../analyse/第一分类下不同类别折扣商品中位数统计.png')
# test=sns.regplot(num_series)
# plt.show()
df=pd.DataFrame({'category':num_series.index,'amount':num_series.values})
# list=num_series.values
# print(list)

totalPrice=[]
index=set(data_2['level_1'])
pNameList=list(index)
for i in pNameList:
    sub=data_2.loc[data_2['level_1']==i]['PricePerItem']
    sub=list(sub)
    totalPrice.append(sub)

plt.xlim(xmax=45)
plt.xticks(np.arange(0,36,3))
plt.ylabel('分布密度',fontproperties=font,fontsize=14)
plt.xlabel('价格区间',fontproperties=font,fontsize=14)
plt.title('第一分类下不同类别商品价格数据密度函数',fontproperties=font,fontsize=20)

Colors=['black','red','green','purple','blue','brown','grey','yellow','orange','pink','red']

for i in range(len(totalPrice)):
    sns.kdeplot(totalPrice[i],color=Colors[i])
plt.savefig('../analyse/第一分类下不同类别商品价格数据密度函数.png')
plt.show()

