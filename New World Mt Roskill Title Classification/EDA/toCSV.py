import pandas as pd

category=[]
pName=[]
with open('../Data/Data_afterEDA2.txt','r',encoding='UTF-8') as file:
    for line in file.readlines():
        line=line.strip()
        category.append(line.split('\t')[0])
        pName.append(line.split('\t')[1])

df=pd.DataFrame({'pName':pName,'Cat_l1':category})
# print(df.head())
df.to_csv('../Data/Data_afterEDA2.csv',index=False,encoding='UTF-8')



