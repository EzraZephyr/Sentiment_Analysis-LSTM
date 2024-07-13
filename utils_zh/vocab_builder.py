import pickle
from nltk.tokenize import word_tokenize
from tensorflow import keras
from sklearn.model_selection import train_test_split

def vocab_builder():
    with open('./preprocessed_data/x.pkl', 'rb') as f:
        x = pickle.load(f)
    with open('./preprocessed_data/y.pkl', 'rb') as f:
        y = pickle.load(f)

    index_to_word, all_words = ['PAD',],[]
    # 初始化词汇表和单词表 因为后续要将不满100个单词的句子用0填充
    # 所以词汇表的第一位‘0’ 为填充‘PAD’

    i = 0
    for line in x:
        i+=1
        if i % 1000 == 0:
            print(i)
        # 看看遍历到哪了

        words = word_tokenize(line)
        # 对每一行文本进行分词

        all_words.append(words)
        # 将每一份的文本分词添加到单词表中

        for word in words:
            if word not in index_to_word:
                index_to_word.append(word)
                # 构建唯一单词的词汇表 用于将索引转换为单词

    word_to_index = {word:idx for idx,word in enumerate(index_to_word)}
    # 反过来构建一下词汇对索引的表示 方便将词汇转换为索引

    word_count = len(index_to_word)

    corpus_idx = []
    # 用于存放每句话转换完索引之后的数据

    for line in all_words:
        temp = []
        for word in line:
            temp.append(word_to_index[word])
            # 这是完整的一段文本 也就是一条评论的索引数据

        temp = keras.preprocessing.sequence.pad_sequences([temp], maxlen=100, padding='post')
        # maxlen=100 将超过100个词汇的评论裁剪为100
        # padding='post' 不够100条的在词尾填充‘0’

        corpus_idx.append(temp[0])
        # 因为pad_sequences返回的是二维数组 所以这里要用temp[0]来提取内容
        # 否则corpus_idx 将变成一个三维数组

    label_to_index = {"positive": 1, "negative": 0}
    y_converted = [label_to_index[label] for label in y]
    # 构建一个字典 将目标值转换为1或0

    X_train, X_test, y_train, y_test = train_test_split(corpus_idx,y_converted, test_size=0.2)
    # 划分训练集和测试集 8:2


    with open('./preprocessed_data/index_to_word.pkl', 'wb') as f:
        pickle.dump(index_to_word, f)
    with open('./preprocessed_data/word_to_index.pkl', 'wb') as f:
        pickle.dump(word_to_index, f)
    with open('./preprocessed_data/word_count.pkl', 'wb') as f:
        pickle.dump(word_count, f)
    with open('./preprocessed_data/corpus_idx.pkl', 'wb') as f:
        pickle.dump(corpus_idx, f)
    with open('./preprocessed_data/y_converted.pkl', 'wb') as f:
        pickle.dump(y_converted,f)
    with open('./preprocessed_data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train,f)
    with open('./preprocessed_data/X_test.pkl', 'wb') as f:
        pickle.dump(X_test,f)
    with open('./preprocessed_data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train,f)
    with open('./preprocessed_data/y_test.pkl', 'wb') as f:
        pickle.dump(y_test,f)
    # 保存数据
