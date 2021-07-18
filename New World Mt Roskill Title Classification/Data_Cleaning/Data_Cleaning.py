import pandas as pd
import numpy as np

'''读取原始数据文件'''
RawData = pd.read_csv('../Data/RawData.csv')
# col_list = ['PID','Pname','price','price_item','url','cat1','cat2','cat_url','store_name']

'''去掉相同&无用项'''
DropList = ['Branch', 'PriceMode', 'ProductId','PromoBadgeImageLabel','Index']
Data_Usefull = RawData.drop(DropList, axis=1)  # 去掉相同&无用项
# print(Data_Usefull.shape)

''' 去除相同level_3类型的重复产品数据（同名、同价格、同level_3类型）从14612条数据精简为13250条数据'''
level_3_Sets = list(set(Data_Usefull['level_3']))
level_3_Sets.sort()
# print(level_3_Sets)
data_1 = pd.DataFrame()
for levelName in level_3_Sets:
    sub = Data_Usefull.loc[Data_Usefull['level_3'] == levelName]
    sub = sub.drop_duplicates(subset=['ProductName', 'PricePerItem'], keep='first')
    data_1 = data_1.append(sub)
# print(data_1.shape)
# print(data_1.info())

'''在已经去除重复商品的数据集（data_1）上继续去除单件促销商品（商品名、类目相同，单价不同）将其置于data_2中  经过此过程，数据量从 13250 条降至 12386 条'''
data_2 = pd.DataFrame()
for levelName in level_3_Sets:
    sub = data_1.loc[data_1['level_3'] == levelName]
    #按商品名为排序主要关键字，升序，价格为次要关键字，降序
    sub = sub.sort_values(by=['ProductName', 'PricePerItem'],ascending=[True,False])
    sub = sub.drop_duplicates(subset=['ProductName'], keep='first')
    data_2 = data_2.append(sub)
# print(data_2.shape)


'''数据分析'''
# #第一类别商品数量
# print(data_2['level_1'].value_counts())
# #第二类别商品数量
# print(data_2['level_2'].value_counts())
# #第三类别商品数量
# print(data_2['level_3'].value_counts())
#
# #第一类别商品价格均值
# print(data_2.groupby('level_1')['PricePerItem'].agg(np.mean))
# #第二类别商品价格均值
# print(data_2.groupby('level_2')['PricePerItem'].agg(np.mean))
# #第三类别商品价格均值
# print(data_2.groupby('level_3')['PricePerItem'].agg(np.mean))
#
# #第一类别商品价格方差
# data_2.groupby(['level_1'])['PricePerItem'].agg(np.var)
# #第二类别商品价格方差
# data_2.groupby(['level_2'])['PricePerItem'].agg(np.var)
# #第三类别商品价格方差
# data_2.groupby(['level_3'])['PricePerItem'].agg(np.var)
#
# #第一类别商品各项数据统计
# print(data_2.groupby(['level_1'])['PricePerItem'].describe())
# #第二类别商品各项数据统计
# print(data_2.groupby(['level_2'])['PricePerItem'].describe())
# #第三类别商品各项数据统计
# print(data_2.groupby(['level_3'])['PricePerItem'].describe())
#
# #商品名 词频统计(前25)
# print(data_2.ProductName.str.split(expand=True).stack().value_counts().head(25))

#打折商品折扣统计
Multi_analyse=data_2.loc[data_2['MultiBuyQuantity']>1]
Multi_analyse=Multi_analyse[['ProductName','MultiBuyQuantity','MultiBuyPrice','PricePerItem','level_1','level_2','level_3']]
Multi_analyse['disCount']=Multi_analyse['MultiBuyPrice']/Multi_analyse['MultiBuyQuantity']/Multi_analyse['PricePerItem']
Multi_analyse.sort_values(by=['disCount'])
Multi_analyse=Multi_analyse[['ProductName','disCount','level_1','level_2','level_3']]


# #第一级分类下折扣商品平均折扣统计
# print(Multi_analyse.groupby(['level_1'])['disCount'].agg(np.mean))
# #第二级分类下折扣商品平均折扣统计
# print(Multi_analyse.groupby(['level_2'])['disCount'].agg(np.mean))
# #第三级分类下折扣商品平均折扣统计
# print(Multi_analyse.groupby(['level_3'])['disCount'].agg(np.mean))

# '''生成最终训练数据文件 置于Data/Data_cleaned.csv''
# #商品名 第一级属性
# col=['PName','Cat_l1']
# outPutCsv=data_2[['ProductName','level_1']]
# outPutCsv.columns=col
# print(outPutCsv.head())
# outPutCsv.to_csv('../Data/Data_cleaned.csv',index=0)
#
# #为EDA准备数据
#
# with open('../Data/Data_forEDA.txt', "w+") as file:
#     for index, row in outPutCsv.iterrows():
#         text_line = row["PName"] + '\t' + row["Cat_l1"]
#         file.writelines(text_line+"\n")
#
