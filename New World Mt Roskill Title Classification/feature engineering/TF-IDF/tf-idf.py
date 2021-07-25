import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


''' 加载数据'''
def loadData():
    data=pd.read_csv('../../Data/Data_cleaned.csv',encoding='UTF-8')
    #缺失值置为-1
    data=data.fillna(-1)
    #pName:商品名
    pName=data.PName.values
    # category_text:分类名
    category_text=data.Cat_l1.values
    #分类：序号
    category2index={c:i for i,c in enumerate(set(category_text))}
    # 序号：分类
    index2category={i:c for i,c in enumerate(set(category_text))}
    #序号化后的分类数据
    category_index=[category2index[i] for i in category_text]
    #返回 商品名、序号化后的分类、分类序号对应的分类名称 列表
    return pName,category_index,index2category


#文本预处理，去除标点和停用词，并均一字母为小写
def text_preprocess(text):
    text=str(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'', '/']
    text = "".join([(a if a not in english_punctuations else " ") for a in text])
    text = " ".join(nltk.tokenize.word_tokenize(text.lower()))
    return text


if __name__ =='__main__':
    pName,category_index,index2category=loadData()
    #净化商品名
    pName=[text_preprocess(i) for i in pName]

    #TD-IDF
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(pName)
    print(features)
