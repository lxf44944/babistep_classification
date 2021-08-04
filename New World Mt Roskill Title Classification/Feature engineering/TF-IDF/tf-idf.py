import warnings

import nltk
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb


warnings.filterwarnings("ignore")
''' 加载数据'''


def loadData():
    print("data loading...\n")
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


# evaluate
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

# TD-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(pName)
# print(features)
# print(index2category)

# '''standard transform'''
# std=StandardScaler()
# features=std.fit_transform(features)

'''start training'''
# quarter for  test
x_train, X_test, y_train, Y_test = train_test_split(features, category_index, random_state=1, test_size=0.25)


def LR():
    """Try LR
    f1:0.97"""
    LR = LogisticRegression(tol=1e-10,n_jobs=8)
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
    """
    knn k=4 acc=0.96 f1:0.96

    """
    knn=KNeighborsClassifier(n_neighbors=4)
    # param_dict={'n_neighbors': [4,5,6,8,9,10,13,15,17]}
    # knn=GridSearchCV(knn, param_grid=param_dict, cv=5)

    knn.fit(x_train, y_train)
    result = knn.predict(X_test)
    evaluation(result, Y_test, index2category, "KNN")



    # print("最佳参数：", knn.best_params_)
    # print("最佳结果：", knn.best_score_)
    # print("最佳估计器", knn.best_estimator_)

    # train_sizes, train_score, test_score = learning_curve(knn, features, category_index,
    #                                                       train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=5,
    #                                                       scoring='accuracy',n_jobs=16)
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('traing examples')
    # plt.ylabel('error')
    # # plt.savefig('./learning_curve.png')
    # plt.show()

def RandomForest():
    #f1:0.971


    rf=RandomForestClassifier(n_estimators=800,n_jobs=16)
    rf.fit(x_train,y_train)
    result=rf.predict(X_test)
    evaluation(result,Y_test,index2category,"RandomForest")

    # train_sizes, train_score, test_score = learning_curve(rf, features, category_index,
    #                                                       train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=5,
    #                                                       scoring='accuracy',n_jobs=16)
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('traing examples')
    # plt.ylabel('error')
    # # plt.savefig('./learning_curve.png')
    # plt.show()

def SVM():
    # undersampled
    #f1:0.985
    #

    model = OneVsRestClassifier(SVC(C=1000, gamma=0.001, kernel='rbf'))
    model.fit(x_train, y_train)
    svm_predict_labels = model.predict(X_test)
    evaluation(svm_predict_labels, Y_test, index2category, "svm")




def lgb_model(x_train, x_test, y_train, y_test, verbose):
    #f1:0.90
    #auc :0.89
    params = {'num_leaves': 60,
              'min_data_in_leaf': 30,
              'objective': 'multiclass',
              'num_class': 33,
              'max_depth': 7,
              'learning_rate': 0.03,
              "min_sum_hessian_in_leaf": 6,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 11,
              "lambda_l1": 0.1,
              "verbosity": -1,
              "nthread": 15,
              'metric': 'multi_error',
              "random_state": 2020
              }

    evals_result = {}
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)
    model = lgb.train(params
                      , lgb_train
                      , num_boost_round=100000
                      , valid_sets=[lgb_train, lgb_test]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , evals_result=evals_result
                      )

    print('Predicting...')
    y_prob = model.predict(x_test, num_iteration=model.best_iteration)
    lgb_predict_labels = [list(x).index(max(x)) for x in y_prob]
    print("AUC score: {:<8.5f}".format(metrics.accuracy_score(lgb_predict_labels, y_test)))
    report = metrics.classification_report(y_test, lgb_predict_labels,
                                           target_names=[index2category[i] for i in range(len(index2category))])
    print(report)
    return model, evals_result




if __name__ == '__main__':
    # model, evals_result = lgb_model(x_train, X_test, y_train, Y_test, verbose=False)
    # lgb.plot_metric(evals_result)
    # plt.show()
    # SVM()
    # RandomForest()
    KNN()
    print('\n')
    # LR()
