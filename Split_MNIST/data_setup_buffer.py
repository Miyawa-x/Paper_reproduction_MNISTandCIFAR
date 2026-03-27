import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def get_split_mnist_loaders(task_id, batch_size=64):
    transform = transforms.ToTensor()
    
    #两个类别
    class_0 = task_id * 2
    class_1 = task_id * 2 + 1

    train_dataset = datasets.MNIST(root='./data_split', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data_split', train=False, download=True, transform=transform)

    def filter_and_remap(dataset, c0, c1):

        idx = (dataset.targets == c0) | (dataset.targets == c1)
        data = dataset.data[idx]
        targets = dataset.targets[idx]

        data = data.unsqueeze(1)
        data = data.float() / 255.0

        targets = torch.where(targets == c0,0,1)

        return TensorDataset(data, targets)
        pass

    # 调用过滤函数
    task_train_dataset = filter_and_remap(train_dataset, class_0, class_1)
    task_test_dataset = filter_and_remap(test_dataset, class_0, class_1)

    train_loader = DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, task_train_dataset

class MemoryBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer_data = []    # [task0_tensor, task1_tensor, ...]
        self.buffer_labels = []  # [task0_labels, task1_labels, ...]
        self.task_sizes = []     # buffer 中保留样本数

    def update_buffer(self, task_dataset, task_id):
        num_tasks_seen = task_id + 1
        
        samples_per_task = self.max_size//num_tasks_seen

        # 修剪旧任务
        if task_id > 0:
            for i in range(len(self.buffer_data)):
                self.buffer_data[i] = self.buffer_data[i][:samples_per_task]
                self.buffer_labels[i] = self.buffer_labels[i][:samples_per_task]
                
        new_data = task_dataset.tensors[0]
        new_labels = task_dataset.tensors[1]

        n_samples = new_data.shape[0] 
        idx = torch.randperm(n_samples)[:samples_per_task]

        sampled_data = new_data[idx]
        sampled_labels = new_labels[idx]

        self.buffer_data.append(sampled_data)
        self.buffer_labels.append(sampled_labels)
        
        print(f"总任务数: {num_tasks_seen}, 当前 Buffer 总大小: {sum([len(d) for d in self.buffer_data])}")

    def get_buffer_data(self):
        if len(self.buffer_data) == 0:
            return None, None
        
        data = torch.cat(self.buffer_data)
        labels = torch.cat(self.buffer_labels)
        return data,labels