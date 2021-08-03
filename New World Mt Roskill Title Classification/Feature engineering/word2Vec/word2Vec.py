import os
import nltk
import numpy as np
import pandas as pd
import torch
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torchtext import vocab
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

''' 加载数据'''


def loadData():
    data = pd.read_csv('../../Data/Data_cleaned_afterEDA2.csv', encoding='UTF-8')
    # 缺失值置为-1
    data = data.fillna(-1)
    # pName:商品名
    pName = data.pName.values
    # category_text:分类名
    category_text = data.Cat_l1.values
    # 分类：序号
    category2index = {c: i for i, c in enumerate(set(category_text))}
    # 序号：分类
    index2category = {i: c for i, c in enumerate(set(category_text))}
    # 序号化后的分类数据
    category_index = [category2index[i] for i in category_text]
    # 返回 商品名、序号化后的分类、分类序号对应的分类名称 列表
    return pName, category_index, index2category


# 文本预处理，去除标点和停用词，并均一字母为小写
def text_preprocess(text):
    text = str(text)
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
    df = pd.read_csv('../../Data/Data_cleaned_afterEDA.csv', encoding='UTF-8')
    tmp = df['pName'].apply(text_preprocess)
    newdf = pd.DataFrame()
    newdf['pName'] = tmp
    # 分类名也需要文本预处理，否则后续向量中会出现 '&' 等 不期望的符号数据
    newdf['Cat_l1'] = df['Cat_l1'].apply(text_preprocess)
    # print(newdf)
    with open('./Data_formatted.txt', 'w+') as newFile:
        for index, value in newdf.iterrows():
            newFile.writelines(value['pName'] + ' ' + value['Cat_l1'] + '\n')


def encode_text_to_features(vector, text):
    vectors = vector.get_vecs_by_tokens(text.split())
    sentence_vector = torch.mean(vectors, dim=0)
    return sentence_vector.tolist()


def w2v_build():
    file = open('Data_formatted.txt', 'r', encoding='UTF-8')
    model = gensim.models.word2vec.Word2Vec(LineSentence(file),
                                            vector_size=300, window=5, min_count=10, sample=1e-5,
                                            workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(r'./word2vec_300dim.txt', binary=False)
    file.close()


'''evaluate'''


def evaluation(predictions, labels, id2label, model_name=None):
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    report = metrics.classification_report(labels, predictions,
                                           target_names=[id2label[i] for i in range(len(id2label))])
    info = "acc:%s, recall:%s, f1 score:%s" % (acc, recall, f1)
    if model_name is not None:
        info = "%s: %s" % (model_name, info)
    print(info)
    print(report)


pName, category_index, index2category = loadData()
# 净化商品名
pName = [text_preprocess(i) for i in pName]
Data_formate()
w2v_build()
vector = load_embeddings()
features = [encode_text_to_features(vector, name) for name in pName]
# print(features)

'''standard transform'''
stdTrans = StandardScaler()
features = stdTrans.fit_transform(features)

x_train, X_test, y_train, Y_test = train_test_split(features, category_index, random_state=1, test_size=0.25)


def LR():
    """LR f1:0.20
        after std:0.16
    """
    LR = LogisticRegression()
    LR.fit(x_train, y_train)

    result = LR.predict(X_test)
    # 结果取整
    result = [np.round(i) for i in result]
    # 范围约束 共11种类型，故结果编号应为 0-10
    upperBound = len(index2category) - 1
    lowerBound = 0
    result = [i if i <= upperBound else upperBound for i in result]
    result = [i if i >= lowerBound else lowerBound for i in result]
    evaluation(result, Y_test, index2category, 'LogisticRegression')



def KNN():
    """knn f1:0.79
           after std:0.79
           k=5 f1:0.84:
    """
    # param_dict = {'n_neighbors': [1,2,3,4,5,6,7]}
    knn = KNeighborsClassifier(n_neighbors=5)
    # knn = GridSearchCV(knn, param_grid=param_dict, cv=10)

    knn.fit(x_train, y_train)
    result = knn.predict(X_test)
    evaluation(result, Y_test, index2category, "KNN")

    # print("最佳参数：", knn.best_params_)
    # print("最佳结果：", knn.best_score_)
    # print("最佳估计器", knn.best_estimator_)
    # train_sizes, train_score, test_score = learning_curve(knn, features, category_index,
    #                                                       train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=5,
    #                                                       scoring='accuracy',n_jobs=8)
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('traing examples')
    # plt.ylabel('error')
    # plt.savefig('./learning_curve.png')

def RandomForest():
    rf=RandomForestClassifier(n_estimators=800,n_jobs=16)
    rf.fit(x_train,y_train)
    result=rf.predict(X_test)
    evaluation(result,Y_test,index2category,"RandomForest")

if __name__ == '__main__':
    # KNN()
    # print('\n')
    # LR()
    RandomForest()