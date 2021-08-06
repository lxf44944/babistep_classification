# New World Mt Roskill Title Classification

# 1. 数据探索 & 数据处理

### 1.1 原始数据格式

数据文件：Data/

代码：Data_Cleaning/**Data_Cleaning.ipynb**

```python
RawData = pd.read_csv('../Data/RawData.csv')
display(RawData.head())
print(RawData.shape)
```

<img src="file:///Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-11-39-59-image.png" title="" alt="" width="680">

原始数据：14612 条样本，15列属性 

### 1.2 缺失值

原始数据中 MultiBuyDeal、MultiBuyQuantity、MultiBuyPrice、PromoBadgeImageLabel  属性列存在较多的缺失值

### 1.3 数据处理

**去除无关项**  得到精简后的数据集：Data_Usefull

```python
'''去掉相同&无用项'''
DropList = ['Branch', 'PriceMode', 'ProductId','PromoBadgeImageLabel','Index']
Data_Usefull = RawData.drop(DropList, axis=1)  # 去掉相同&无用项
Data_Usefull.shape
(14612, 10)
```

**去除重复商品数据** 继续从Data_Usefull基础上进行去重得到临时数据集data_1

经过此过程，数据量从 14612 条降至 13250 条

```python
''' 去除相同level_3类型的重复产品数据
（同名、同价格、同level_3类型）从14612条数据精简为13250条数据'''

level_3_Sets = list(set(Data_Usefull['level_3']))
level_3_Sets.sort()
# print(level_3_Sets)
data_1 = pd.DataFrame()
for levelName in level_3_Sets:
    sub = Data_Usefull.loc[Data_Usefull['level_3'] == levelName]
    sub = sub.drop_duplicates(subset=['ProductName', 'PricePerItem'], keep='first')
    data_1 = data_1.append(sub)
data_1.shape
data_1.info()
```

在data_1基础上**去除单件折扣商品** 得到 data_2  经过此过程，数据量从 13250 条降至 12386 条

```python
data_2 = pd.DataFrame()
for levelName in level_3_Sets:
    sub = data_1.loc[data_1['level_3'] == levelName]
    #按商品名为排序主要关键字，升序，价格为次要关键字，降序
    sub = sub.sort_values(by=['ProductName', 'PricePerItem'],ascending=[True,False])
    sub = sub.drop_duplicates(subset=['ProductName'], keep='first')
    data_2 = data_2.append(sub)
data_2.shape
display(data_2.head())
```

**多件促销商品的折扣统计** 得到分析数据集 Multi_analyse

```python
'''折扣商品信息处理'''
Multi_analyse=data_2.loc[data_2['MultiBuyQuantity']>1]
Multi_analyse=Multi_analyse[['ProductName','MultiBuyQuantity','MultiBuyPrice','PricePerItem','level_1','level_2','level_3']]
Multi_analyse['disCount']=Multi_analyse['MultiBuyPrice']/Multi_analyse['MultiBuyQuantity']/Multi_analyse['PricePerItem']
Multi_analyse.sort_values(by=['disCount'])
Multi_analyse=Multi_analyse[['ProductName','disCount','level_1','level_2','level_3']]
```

## 1.4 数据分析

分析代码：Data_Cleaning/**Data_Cleaning.ipynb**

绘图代码：Data_Cleaning/**analyse.py**

统计图片：Analyse/

**第一类别商品各项数据统计**

```python
#第一类别商品各项数据统计
data_2.groupby(['level_1'])['PricePerItem'].describe()
```

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-12-31-27-image.png)

****第一分类下不同类别商品数量统计****

![第一分类下不同类别商品数量统计.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别商品数量统计.png)

****第一分类下不同类别商品均价统计****

![第一分类下不同类别商品均价统计.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别商品均价统计.png)

**第一分类下不同类别商品商品价格方差统计**

![第一分类下不同类别商品商品价格方差统计.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别商品商品价格方差统计.png)

**第一分类下不同类别折扣商品中位数统计**

![第一分类下不同类别折扣商品中位数统计.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别折扣商品中位数统计.png)

**第一分类下不同类别商品价格数据密度函数**

![第一分类下不同类别商品价格数据密度函数.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别商品价格数据密度函数.png)

**第一分类下不同类别商品价格数据正态拟合分布**

![第一分类下不同类别商品价格数据正态拟合分布.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/第一分类下不同类别商品价格数据正态拟合分布.png)

从上述对第一级类别商品的分析结果来看，不同类别商品数量差距较大，Pantry 类样本数量超过3500 而 Featured 类 数量不足500 。分布情况不利于后续学习过程，鉴于最少数量的类别数量过小，因此在后续应该对样本进行过采样扩充数据，而后进行欠采样以平衡各类别样本量。另，从特征选择的角度看。不同类别间商品价格分布重合较多，差距较小，比较难以区分，切折扣商品存在较多缺失值。因此从价格角度不利于特征的提取。后续选择从商品名角度利用文本处理的一些算法进行特征值的处理。

### 1.5 数据保存

最后仅保留商品名和第一级分类类别 两列属性。将初步处理后的数据集转存为 Data目录下**Data_cleaned.csv** 文件

```python
'''生成最终训练数据文件 置于Data/Data_cleaned.csv'''
#商品名 第一级属性
col=['PName','Cat_l1']
outPutCsv=data_2[['ProductName','level_1']]
outPutCsv.columns=col
print(outPutCsv.head())
outPutCsv.to_csv('../Data/Data_cleaned.csv',index=0)
```

### 1.6 EDA 文本增强

通过EDA方法，希望增加过少样本量类别商品的数量。

EDA算法使用项目中eda_nlp包中的相关方法。

Data_Cleaning/**Data_Cleaning.ipynb:**

```python
'''为EDA准备数据'''
with open('../Data/Data_forEDA.txt', "w+") as file:
    for index, row in outPutCsv.iterrows():
        text_line = row["Cat_l1"] + '\t' + row["PName"]
        file.writelines(text_line+"\n")为
```

为EDA准备训练数据 样本标签在前，名称在后，分行写入**Data_forEDA.txt**文件中

将原始中的训练数据增加13倍，每条样本中数据改变量10%，生成强化后的文件 **Data_afterEDA.txt**。 而后通过*EDA/**toCSV.py** 中的方法，将txt结果文件重新转化为csv文件得到 **Data_afterEDA.csv**

```py
python code/augment.py --input=../../Data/Data_forEDA.txt --output=../../Data/Data_afterEDA.txt --num_aug=13 --alpha=0.05
```

通过EDA/**analyse_afterEDA.py**  对增强后的结果进行分析

```python
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
```

![EDA.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/EDA.png)

<img src="file:///Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-13-29-26-image.png" title="" alt="" width="542">

根据上图，发现原数据中数量偏低的几个分类，在经过EDA的强化处理后，均有了超过1000

的样本数量。但不同分类间的数量比例依然严重失衡，需要进一步对其进行平衡化处理。

接下来进行数据欠采样。通过**imblearn**包中**RandomUnderSampler** 方法，对过多的数据进行随机删除完成样本数量平衡。相关方法位于 EDA/**Data_Cleaned_afterEDA.py**文件中

首先对样本进行重新去重操作

```python
#load Data
data=pd.read_csv('../Data/Data_afterEDA.csv')

#drop same data
data=data.drop_duplicates(subset=['pName'],keep='first')
pName=pd.DataFrame(data['pName'])
Cat_l1=pd.DataFrame(data['Cat_l1'])
```

样本数量平衡处理 undersampling ，结果存为 Data_cleaned_afterEDA.csv

```python
rus = RandomUnderSampler()
X_resampled, y_resampled = rus.fit_resample(pName, Cat_l1)
print(y_resampled.value_counts())

df=pd.DataFrame({'pName':X_resampled.pName,'Cat_l1':y_resampled.Cat_l1})
df.to_csv('../Data/Data_cleaned_afterEDA.csv',index=False,encoding='UTF-8')
```

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-13-37-20-image.png)

经过欠采样处理，各类别样本数量已均衡，均达到1147 样本量

# 2. 特征工程

### 2.1 准备工作

loadData 函数，载入数据集，并建立 序号：类别名 、类别名：序号 映射集 

返回 商品名(pName)、序号化后的分类(category_index)、分类序号对应的分类名称(index2category)列表

```python
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
```

text_preprocess 函数集成了对商品名的去标点、统一小写、去停用词等操作。

用**nltk.tokenize**包中的相关方法进行停用词去除

```python
def text_preprocess(text):
    text = str(text)
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'', '/']
    text = "".join([(a if a not in english_punctuations else " ") for a in text])
    text = " ".join(nltk.tokenize.word_tokenize(text.lower()))
    return text
```

### 2.2 TF-IDF

对应模块：Feature engineering/TF-IDF/**tf-idf.py**

TF-IDF 算法运用**sklearn.feature_extraction.text** 中**TfidfVectorizer** 模块，得到特征集

features

```python
# TD-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(pName)　
```

### 2.3 Word2Vec

对应模块：Feature engineering/word2Vec/**word2Vec.py**

数据格式化，空格相隔按行写入 得到**Data_formatted.txt**

```python
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
```

转换函数

使用**gensim**函数库训练Word2Vec模型，将每条样本转化成维度为300的词向量，存储为**word2vec_300dim.txt**

```python
def w2v_build():
    file = open('Data_formatted.txt', 'r', encoding='UTF-8')
    model = gensim.models.word2vec.Word2Vec(LineSentence(file),
                                            vector_size=300, window=5, min_count=10, sample=1e-5,
                                            workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(r'./word2vec_300dim.txt', binary=False)
    file.close()
```

得到模型和特征转换

```python
def load_embeddings():
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    word2vec = vocab.Vectors(name=r'./word2vec_300dim.txt', cache=cache)
    return word2vec


def encode_text_to_features(vector, text):
    vectors = vector.get_vecs_by_tokens(text.split())
    sentence_vector = torch.mean(vectors, dim=0)
    return sentence_vector.tolist()
```

```python
pName, category_index, index2category = loadData()
# 净化商品名
pName = [text_preprocess(i) for i in pName]
Data_formate()
w2v_build()
vector = load_embeddings()
features = [encode_text_to_features(vector, name) for name in pName]
```

最后得到特征集 features

### 2.4 标准化及训练、测试集划分

使用sklearn.preprocessing中StandardScale进行数据标准化

使用sklearn.model_selection中train_test_split进行数据集划分，训练集测试集按 3:1 划分

```python
stdTrans = StandardScaler()
features = stdTrans.fit_transform(features)

x_train, X_test, y_train, Y_test = train_test_split(features, category_index, random_state=1, test_size=0.25)
```

# 3.模型建立

### **3.1 LogisticRegression**

```python
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
```

**TF-IDF**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-14-59-50-image.png)

**Word2Vec**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-01-20-image.png)

### 3.2 KNeighbors

```python
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
```

**TF-IDF**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-03-30-image.png)

**Word2Vec**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-05-22-image.png)

### 3.3 RandomForest

```python
def RandomForest():

    # 800 f1:0.947

    rf=RandomForestClassifier(n_estimators=1000,n_jobs=16)
    rf.fit(x_train,y_train)
    result=rf.predict(X_test)
    evaluation(result,Y_test,index2category,"RandomForest")
```

**TF-IDF**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-14-10-image.png)

**Word2Vec**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-12-48-image.png)

## 

### 3.4 lightgbm

```python
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
```

**TF-IDF**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-17-18-image.png)

### 3.5 SVM

```python
def SVM():

    model = OneVsRestClassifier(SVC(C=1, gamma=20, decision_function_shape='ovr'))
    model.fit(x_train, y_train)
    svm_predict_labels = model.predict(X_test)
    evaluation(svm_predict_labels, Y_test, index2category, "svm")
```

**TF-IDF**

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-20-27-image.png)

# 

# 4.模型优化

### 4.1 学习曲线

几个简单模型的学习曲线均类似下图

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-22-38-image.png)

![tfidf_lgb_curve.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/tfidf_lgb_curve.png)

从以上简单模型测试结果可以得出 整体上TF-IDF 算法获取的特征值在学习中相比Word2Vec算法有更好的效果。但从学习曲线趋势上看，整体还有优化空间，样本不足限制了测试曲线的进一步下降。所以后续应再次对样本数量进行加强以获得更好结果

### 4.2 GridSearchCV 超参数选择

使用**sklearn.model_selection** 中 **GridSearchCV** 模块进行参数试探 ，cross validated 交叉验证折数选择5折  

**KNN**        

```python
    knn=KNeighborsClassifier(n_neighbors=4)
    param_dict={'n_neighbors': [2,3,4,5,6,8,9,10,13,15,17]}
    knn=GridSearchCV(knn, param_grid=param_dict, cv=5)

    knn.fit(x_train, y_train)
    # result = knn.predict(X_test)
    # evaluation(result, Y_test, index2category, "KNN")



    print("最佳参数：", knn.best_params_)
    print("最佳结果：", knn.best_score_)
    print("最佳估计器", knn.best_estimator_)
```

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-38-08-image.png)

**SVM**

```python
uned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()

means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("Detailed classification report:")
    print('\n')
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
```

最佳结果

```python
{
    'C':1000, 'gamma':0.001, 'kernel'='rbf'
}
```

### 4.3 样本扩充

针对之前分析样本不足的情况，继续使用EDA方法扩充样本，将样本扩充 100 倍 ，同样对EDA 处理后的数据进行清理和平衡操作 最终获得数据集 **Data_afterEDA2.csv**

```python
python code/augment.py --input=../../Data/Data_forEDA.txt --output=../../Data/Data_afterEDA2.txt --num_aug=100 --alpha=0.05
```

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-55-20-image.png)

平衡后各分类数据量相比原来扩充近4倍

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-15-57-19-image.png)

![img.png](/Users/lishengdi/Job/babistep_classification/New%20World%20Mt%20Roskill%20Title%20Classification/Analyse/img.png)

扩充后提升效果明显,同KNN算法 F1 分数由原来的 0.885 提升至 0.962 ，二次加强前预测结果相对较差的Pantry 类经过这轮扩充，F1从原先0.79 提升至 0.90。其他模型也均有很大程度的提升学习曲线也有所下降

# 5. 评估对比

### 5.1结果对比

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-16-42-11-image.png)

### 5.2 模型保存

![](/Users/lishengdi/Library/Application%20Support/marktext/images/2021-08-06-16-27-57-image.png)

经过对比，最终选择TF-IDF算法下的SVM 模型，并将其通过 **joblib**包中的方法保存为.model 格式文件:**SVM_TFIDF_classifier.model**

```python
    svm = OneVsRestClassifier(SVC(C=1000, gamma=0.001, kernel='rbf'))
    svm.fit(x_train, y_train)
    joblib.dump(value=svm,filename='../../res/SVM_TFIDF_classifier.model')
```

# 6.目录结构说明

## New World Mt Roskill Title Classification/

### Analyse/   存放数据分析所绘制的图片

### Data/         存放处理所需数据集

### Data_Cleaning/    数据清理、分析相关代码

#### Data_Cleaning.ipynb  数据清理、各项数值统计

#### analyse.py  绘图代码段

### EDA/       EDA 增强后相关操作

#### analyse_afterEDA.py   绘制统计图

#### Data_Cleaninf_afterEDA.py    EDA增强后的数据清理

#### toCSV.py      将EDA结果由txt转化为csv

### Feature_engineering/    特征工程及模型

#### TF-IDF/     tf-idf 算法相关代码及模型

#### word2Vec/    word2Vec算法相关代码、资源及模型

### res/       项目资源包

#### eda_nlp/   EDA 算法包

#### Songti.ttc   pylot 绘图所用字体文件

#### SVM_TFIDF_classifier.model   最终生成的模型文件
