import torch
from torch.utils.data import Dataset

class data_process(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    # 初始化 用于接收数据和目标值

    def __len__(self):
        return len(self.X)
    # 返回数据的长度 方便dataloader根据批量大小进行处理

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)
    # 将数据转化为张量并返回

