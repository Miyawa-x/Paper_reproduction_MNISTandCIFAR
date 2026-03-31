import math
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from data_setup_buffer import get_split_mnist_loaders

class SupersampleDataset(Dataset):
    def __init__(self, task_dataset):
        # 底层数据
        data = task_dataset.tensors[0]    # [N, 1, 28, 28]
        labels = task_dataset.tensors[1]  # [N] (0 或 1)
        
        # 分离数据
        data_c0 = data[labels == 0]
        data_c1 = data[labels == 1]
        
        # 两两配对
        len_c0 = len(data_c0) - (len(data_c0) % 2)
        len_c1 = len(data_c1) - (len(data_c1) % 2)
        data_c0 = data_c0[:len_c0]
        data_c1 = data_c1[:len_c1]
        
        pairs_c0 = data_c0.view(-1,2,1,28,28)
        pairs_c1 = data_c1.view(-1,2,1,28,28)
        
        # 拼接两个类别的配对数据，生成对应的 labels
        self.all_pairs = torch.cat([pairs_c0,pairs_c1],dim = 0)
        labels_c0 = torch.zeros(len(pairs_c0))
        labels_c1 = torch.ones(len(pairs_c1))
        self.labels = torch.cat([labels_c0,labels_c1])

        total_pairs = self.all_pairs.shape[0]

        self.S = torch.randint(0, 2, (total_pairs,))

        indices = torch.arange(total_pairs)

        self.train_data = self.all_pairs[indices, self.S]
        self.test_data = self.all_pairs[indices, 1 - self.S]

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        # 每次吐出一组：训练图，测试图，标签，硬币结果 S
        return self.train_data[idx], self.test_data[idx], self.labels[idx], self.S[idx]

def calculate_mi_and_bounds(model, super_loader, device, n, m=400):
    
    #计算互信息并输出四大理论界限。
    model.eval()
    
    # \Delta 取值域 {-1, 0, 1}, S 取值域 {0, 1}
    joint_counts = {} 
    total_pairs = 0
    
    with torch.no_grad():
        for train_img, test_img, labels, s in super_loader:
            train_img, test_img, labels = train_img.to(device), test_img.to(device), labels.to(device)
            
            train_out = model(train_img)
            test_out = model(test_img)

            _,train_pred = torch.max(train_out,1)
            _,test_pred = torch.max(test_out,1)

            train_loss = (train_pred != labels).float()
            test_loss = (test_pred != labels).float()
            
            delta = test_loss - train_loss
            
            # 将 (\Delta, S) 存入字典进行频率统计
            for d, s_val in zip(delta.cpu().numpy(), s.numpy()):
                key = (d, s_val)
                joint_counts[key] = joint_counts.get(key, 0) + 1
                total_pairs += 1

    # 信息论计算
    mi_delta_s = 0.0 
    if(total_pairs > 0):
        p_delta = {}
        p_s = {}
        for (d,s), cnt in joint_counts.items():
            p_delta[d] = p_delta.get(d,0) + cnt
            p_s[s] = p_s.get(s,0) + cnt

        for k in p_delta: p_delta[k] /= total_pairs
        for k in p_s: p_s[k] /= total_pairs

        for (d, s), cnt in joint_counts.items():
            p_joint = cnt / total_pairs
            pd = p_delta[d]
            ps = p_s[s]
            if pd > 0 and ps > 0 and p_joint > 0:
                mi_delta_s += p_joint * math.log2(p_joint / (pd * ps))

    sq_bound = math.sqrt((2 * mi_delta_s) / m)

    # Binary KL
    bkl_bound = math.sqrt((2 * mi_delta_s * math.log(2)) / m)

    # Weighted Bound
    C1, C2 = 0.1, 0.3
    weighted_bound = C1 * mi_delta_s + C2 * math.sqrt(mi_delta_s / m)

    # Variance Bound
    mean_d = sum(d * p_delta[d] for d in p_delta) if p_delta else 0
    var_d = sum((d - mean_d) ** 2 * p_delta[d] for d in p_delta) if p_delta else 0
    var_bound = math.sqrt((var_d * mi_delta_s) / m) if m > 0 else 0
    
    print(f"Mutual Information I(\Delta; S): {mi_delta_s:.6f}")
    print(f"互信息与理论界限：")
    print(f"I(Δ;S)    : {mi_delta_s:.6f}")
    print(f"SQ Bound  : {sq_bound:.6f}")
    print(f"KL Bound  : {bkl_bound:.6f}")
    print(f"WEI Bound : {weighted_bound:.6f}")
    print(f"VAR Bound : {var_bound:.6f}")
    
    return mi_delta_s


if __name__ == '__main__':
    
    # Task 0 数据
    _, _, raw_dataset = get_split_mnist_loaders(task_id=0, batch_size=64)
    print(f"原始数据总量: {len(raw_dataset)}")
    
    # 实例化
    super_dataset = SupersampleDataset(raw_dataset)
    print(f"生成的超样本对总量: {len(super_dataset)}") 
    
    # 取一组数据检查
    train_img, test_img, label, s = super_dataset[0]
    
    print(f"\n[抽查第一个样本对]")
    print(f"Train Image 形状: {train_img.shape}") 
    print(f"Test Image 形状: {test_img.shape}")   
    print(f"共享标签 (Label): {label.item()}")    
    print(f"抛硬币结果 (S): {s.item()}")          
    
    # 验证 Batch
    super_loader = DataLoader(super_dataset, batch_size=32, shuffle=True)
    batch_train, batch_test, batch_labels, batch_s = next(iter(super_loader))
    print(f"\n[验证 DataLoader]")
    print(f"Batch Train 维度: {batch_train.shape}")
    print(f"Batch S 维度: {batch_s.shape}")