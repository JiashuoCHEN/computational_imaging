import torch
from torchvision import datasets, transforms

# 配置路径
data_root = './data/mnist/rawdata'  # 当前目录下的data/mnist/rawdata
download = True  # 设置为False避免重复下载

# 自动创建目录（若不存在）
import os
os.makedirs(data_root, exist_ok=True)

# 下载并加载数据集
train_dataset = datasets.MNIST(
    root=data_root,
    train=True,
    download=download,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root=data_root,
    train=False,
    download=download,
    transform=transforms.ToTensor()
)

# 验证数据集完整性
print(f"训练集样本数: {len(train_dataset)}")  # 应输出60000
print(f"测试集样本数: {len(test_dataset)}")   # 应输出10000

# 显示第一张训练图像
import matplotlib.pyplot as plt
plt.imshow(train_dataset[0][0].squeeze(), cmap='gray')
plt.title(f"Label: {train_dataset[0][1]}")
plt.show()