import pickle
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils_zh.data_process import data_process
from utils_zh.model import sentimentmodel

def train():
    with open('./preprocessed_data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('./preprocessed_data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('./preprocessed_data/word_count.pkl', 'rb') as f:
        word_count = pickle.load(f)

    train_process = data_process(X_train, y_train)
    # 创建数据处理的对象

    model = sentimentmodel(word_count, 1)
    # 初始化模型 输出设定为一

    dataloader = DataLoader(train_process, batch_size=32, shuffle=True,drop_last=True)
    # 创建数据加载器 每32个为一组 并且最后不足32个的舍弃

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 如果GPU可用的话就在GPU上运行 否则CPU

    criterion = nn.BCELoss()
    # 使用二分类交叉熵损失函数 更好的度量与测概率和目标值之间的误差

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 使用Adam自适应学习率来优化参数更新

    epoch = 35
    # 设置训练轮数

    train_log = './model/training.log'
    file = open(train_log, 'w')
    # 保存训练时的日志（好习惯）

    for epoch in range(epoch):
        epoch_idx = 0
        total_loss = 0.0
        start_time = time.time()
        # epoch_idx用于记录处理了多少个batch 方便求平均损失
        # total_loss用于计算每一次训练的总损失
        # 记录时间 计算每一次训练的时长

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)
            # 将数据和目标值移动到可用设备上

            hidden = model.init_hidden(batch_size=X.size(0))
            # 初始化隐藏状态

            output, hidden = model(X,hidden)
            # 向前传播 得到更新后的输出值和隐藏状态

            output = output.squeeze()
            # 去除多余的维度

            optimizer.zero_grad()
            # 梯度清零 防止梯度累加

            loss = criterion(output, y)
            # 计算损失

            total_loss += loss.item()
            # 将损失累加 注意要加item取出标量值 否则就是一个张量

            loss.backward()
            # 反向传播

            optimizer.step()
            # 更新模型参数

            epoch_idx += 1

        message = 'Epoch:{}, Loss:{:.4f}, Time{:.2f}'.format(epoch+1, total_loss / epoch_idx, time.time() - start_time)
        file.write(message + '\n')
        print(message)
        # 将每一轮训练的信息打印并输出 方便发现训练时的问题

    file.close()
    torch.save(model.state_dict(), './model/sentiment_model.pt')
    # 将日志文件关闭 并且保存训练好的模型


