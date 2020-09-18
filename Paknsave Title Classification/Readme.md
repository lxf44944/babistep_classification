# **Paknsave** 商品类别分类

此实验目的为解决自然语言处理的问题，根据数据集训练出最佳商品分类模型：能利用爬虫所获取包含超市商品与类别类目的数据集，依照商品名称及其对应的商品类别(总共有9种类别) 进行分类。实验从使用基础模型开始，通过对优化以其达到最佳分类效果。
## 项目思路
1. 将原始爬取数据格式化为csv文件，进行数据清洗和探索。
2. 分别选用word2vec和tf-idf进行特征提取，分别使用LR，KNN，SVM，TF-IDF模型进行学习。
3. 选取上述模型中表现较好的模型通过eda数据增强，样本欠采样和网格搜索参数进行进一步优化，找到相对最优模型。
## 文件解释
`Paknsave Title Classification`目录下共有6个文件，分别为
- `raw_data`：包含`Paknsave Title Crawler`项目的爬取结果，是网站 https://www.paknsaveonline.co.nz 中的原始商品数据。
- `data_cleaning`：包含了建模前的数据探索和数据清理，以及后续eda数据增强所需要的数据处理。
    - `csv_convert`：负责读取`raw_data`文件夹中的数据，转化为csv格式，并生成：
        - `raw_data.csv`：转化为csv格式以后的商品原始分类数据。
    - `data_cleaning.ipynb`：读取`raw_data.csv`中原始数据，进行数据清理和数据探索，过程中生成以下文件：
        - `unique_product.csv`： 去除同一分店内重复数据和无关列之后的数据集，共390811条数据。
        - `unique_product_match.csv`：去除同一分店内重复数据和无关列，并与Countdown项目数据保持一致的数据集，共390811条数据。
        - `product_classification.csv`：去除所有重复数据，只保留name和cat信息的数据集，共35789条数据。
        - `train.csv`：在`product_classification.csv`中随机提取75%数据成为训练集（用于eda-undersampling），共8948条数据。
        - `test.csv`：在`product_classification.csv`中随机提取25%数据成为测试集（用于eda-undersampling），共26841条数据。
        - `train.txt`：将`train.csv`转化为txt格式，作为eda增强的输入。
        - `eda_data_augmented.txt`：eda数据增强的输出，将原始数据增加16倍，每条数据改变量为10%，共456397条数据。
        - `eda_data_augmented.csv`：将eda转化为可用的csv格式，共456397条数据。
- `eda_nlp-master`：eda数据增强所使用的包，其中：
    - `command.md`： 记录执行本项目eda数据增强所需的命令行指令。
- `word2vec`：将`product_classification.csv`中的数据用word2vec进行转化，分别使用LR，KNN，SVM，TF-IDF进行学习。
- `tf-idf`：将`product_classification.csv`中的数据用word2vec进行转化，分别使用LR，KNN，SVM，TF-IDF进行学习，并针对表现较好的SVM寻找最优参数。
- `eda-undersampling`：在tf-idf模型的基础上，使用eda增强后的数据`eda_data_augmented.csv`进行学习，对数据进行欠采样处理，用KNN和最优参数的SVM对数据进行建模学习。
## 项目成果
目前为止，项目找到的最佳模型为eda增强，数据欠采样，和网格优化后的SVM模型，模型准确率达97%，召回率和f1-score达到98%，并且学习曲线的方差和偏差都较小，整体结果比较理想。