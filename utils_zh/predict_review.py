import pickle
import keras
import torch
from nltk import word_tokenize
from utils_zh.data_loader import clean_text
from utils_zh.model import sentimentmodel


def load_model():
    with open('./preprocessed_data/word_to_index.pkl','rb') as f:
        word_to_index = pickle.load(f)


    model = sentimentmodel(len(word_to_index),1)
    model.load_state_dict(torch.load('./model/sentiment_model.pt', map_location=torch.device('cpu')))
    model.eval()
    # 设置模型的嵌入词汇量和输出 并且加载模型 设定为评估模式

    return model, word_to_index

def predict_input(model, word_to_index, review):
    sentence = []
    # 用来储存被转换为索引后的句子

    """review = input("Please enter your review (or type 'exit' to quit): ")
    if review.lower() == 'exit':
        break"""
    # 如果不采用GUI界面输入的话 以上为输入方式

    text = word_tokenize(clean_text(review))
    # 将输入后的句子经过正则处理后分词

    for word in text:
        if word in word_to_index:
            sentence.append(word_to_index[word])
            # 如果这个词在词汇表里的话则添加

        else:
            sentence.append(0)
            # 否则置为0

    sentence = keras.preprocessing.sequence.pad_sequences([sentence], maxlen=100, padding='post')
    # 将这组索引裁剪到固定的100长度 不够的用0填充

    sentence = torch.tensor(sentence, dtype=torch.long).unsqueeze(0)
    # 将这组索引转化为扎改良 并且增加一个维度以匹配模型的输入格式

    hidden = model.init_hidden(1)
    output, _ = model(sentence, hidden)
    output = output.squeeze()
    # 训练模型 并删除多余的维度

    answer = (output>=0.5).float()
    # 将答案转为二分类结果

    if answer :
        print('这是一条好评')
        return '这是一条好评'
    else:
        print('这是一条差评')
        return '这是一条差评'
        # return是为了使GUI界面的弹窗接受到返回的字符串并显示
        # 如果不使用GUI的话则可以删除这两个return


"""def predict_review():
    model, word_to_index = load_model()
    predict_input(model, word_to_index)"""
    # 如果不使用GUI界面的话 则使用这函数进行调用即可
