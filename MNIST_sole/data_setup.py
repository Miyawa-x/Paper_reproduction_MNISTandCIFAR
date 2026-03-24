import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    # 定义数据预处理
    transform = transforms.ToTensor()

    # 获取训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 获取测试集
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                download=True, 
                                transform=transform)

    # 封装成 DataLoader 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
