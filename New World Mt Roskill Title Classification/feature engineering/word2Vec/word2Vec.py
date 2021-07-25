import os

import nltk
import numpy as np
import pandas as pd
import torch
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

from torchtext import vocab

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


def load_embeddings():
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    word2vec = vocab.Vectors(name=r'./word2vec_300dim.txt', cache=cache)
    return word2vec

def Data_formate():
    df = pd.read_csv('../../Data/Data_cleaned.csv', encoding='UTF-8')
    tmp = df['PName'].apply(text_preprocess)
    newdf = pd.DataFrame()
    newdf['PName'] = tmp
    #分类名也需要文本预处理，否则后续向量中会出现 '&' 等 不期望的符号数据
    newdf['Cat_l1'] = df['Cat_l1'].apply(text_preprocess)
    print(newdf)
    with open('./Data_formatted.txt', 'w+') as newFile:
        for index, value in newdf.iterrows():
            newFile.writelines(value['PName'] + ' ' + value['Cat_l1'] + '\n')

def encode_text_to_features(vector, text):
    vectors = vector.get_vecs_by_tokens(text.split())
    sentence_vector = torch.mean(vectors, dim=0)
    return sentence_vector.tolist()

def w2v_build():
    file= open('Data_formatted.txt', 'r', encoding='UTF-8')
    model = gensim.models.word2vec.Word2Vec(LineSentence(file),
                    vector_size=300, window=5, min_count=10, sample=1e-5,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(r'./word2vec_300dim.txt', binary=False)

if __name__ =='__main__':
    pName,category_index,index2category=loadData()
    #净化商品名
    pName=[text_preprocess(i) for i in pName]
    Data_formate()
    w2v_build()
    vector = load_embeddings()
    features = [encode_text_to_features(vector, name) for name in pName]
    # print(features)
