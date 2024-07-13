import pickle
import torch
from torch.utils.data import DataLoader
from utils_zh.data_process import data_process
from utils_zh.model import sentimentmodel

def predict():
    word_count, X_test, y_test = load()

    test_process = data_process(X_test, y_test)
    test_loader = DataLoader(test_process, batch_size=32, shuffle=False)
    model = sentimentmodel(word_count, 1)
    model.load_state_dict(torch.load('./model/sentiment_model.pt', map_location=torch.device('cpu')))
    # 加载训练好的模型

    model.eval()
    # 将模型设置为评估模式

    total_correct = 0
    total_count = 0

    with torch.no_grad():
    # 禁用梯度计算 在测试的时候没有必要使用后梯度

        for X, y in test_loader:
            hidden = model.init_hidden(batch_size=X.size(0))
            # 初始化隐藏状态

            outputs, _ = model(X, hidden)
            # 向前传播 且不需要使用隐藏状态 因为不需要向后传播

            outputs = outputs.squeeze()
            # 去除不必要的维度

            y_pred = (outputs >= 0.5).float()
            # 将输出的概率转化为二分类结果 （0或1）

            correct = (y_pred == y).sum().item()
            # 计算预测正确的数量

            total_correct += correct
            total_count += y.size(0)

    accuracy = total_correct / total_count
    print(f'Accuracy: {accuracy*100:.2f}%')
    # 计算并输出准确率

def load():
    with open('./preprocessed_data/word_count.pkl', 'rb') as f:
        word_count = pickle.load(f)
    with open('./preprocessed_data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('./preprocessed_data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return word_count, X_test, y_test