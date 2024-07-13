import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    # 清除内涵的html标签

    text = re.sub(r'[^a-zA-Z]',' ',text)
    # 清楚所有非字母元素

    text = text.lower()
    # 全部转化为小写

    words = word_tokenize(text)
    # 对这一段文本进行分词拆分

    lemmatizer = WordNetLemmatizer()
    # 初始化词形还原器

    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    # 将所有词形进行还原 并同时判断其是否为停用词 不是的话将其储存
    # 停用词为不包含什么实际信息并且出现频率较多的词汇 例如the is a

    text = ' '.join(words)
    # 将处理过的单词进行重新组合

    return text

def dataloader():
    filename = './data/IMDB Dataset.csv'
    data = pd.read_csv(filename)
    # 用pandas取出数据 形成DataFrame形式 便与后续的处理

    x = data['review']
    y = data['sentiment']
    # 取出评论和情感标签

    for i in range(len(x)):
        x[i] = clean_text(x[i])

        if i % 1000 == 0:
            print(i)
        # 便于观测数据处理进度

    with open('preprocessed_data/x.pkl', 'wb') as f:
        pickle.dump(x, f)
    with open('preprocessed_data/y.pkl', 'wb') as f:
        pickle.dump(y, f)
    # 将处理后的数据以二进制形式序列化保存到文件中 便于以后的取用
