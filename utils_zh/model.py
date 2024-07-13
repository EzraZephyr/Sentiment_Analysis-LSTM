import torch
import torch.nn as nn
import torch.nn.functional as F

class sentimentmodel(nn.Module):
    def __init__(self,word_count,output):
        super(sentimentmodel, self).__init__()
        self.ebd = nn.Embedding(word_count,256)
        # 定义词嵌入层 将词汇数量映射为256维的向量

        self.lstm = nn.LSTM(256,256,1,batch_first=True)
        # 定义lstm层 进行一次循环 并且将batch_size置为第一维(好习惯)

        self.out = nn.Linear(256,output)
        #定义全连接层

    def forward(self,X,hidden):
        ebd = self.ebd(X)
        # 首先通过嵌入曾 将词汇索引转换为词向量

        ebd = F.dropout(ebd, p=0.5,training=self.training)
        # 使用dropout使每一次训练时神经元失效一部分 防止过拟合
        # 并且只在训练中生效

        ebd = ebd.squeeze(1)
        # 去除维度为1的维度 简化张量形状并使得其贴合方法的处理

        ebd, hidden = self.lstm(ebd,hidden)
        # 进入LSTM层进行运算

        ebd = ebd[:,-1,:]
        # 取出最后一个时间步的输出

        out = self.out(ebd)
        # 将最后一个时间步通过全连接层得到输出

        return torch.sigmoid(out),hidden
        # 将输出的值映射到0-1之间 方便进行0 1判断

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 256).to(next(self.parameters()).device)
                    ,torch.zeros(1, batch_size, 256).to(next(self.parameters()).device))
        # 初始化并返回隐藏层和细胞层的全0状态 并且使其在同一GPU或CPU上进行计算
