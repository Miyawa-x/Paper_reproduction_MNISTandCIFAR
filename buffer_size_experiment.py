import sys
import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Split_MNIST'))

from data_setup_buffer import get_split_mnist_loaders, MemoryBuffer
from basic_train import train_and_evaluate
from split_model import SimpleNet
from bounds_evaluator import SupersampleDataset, calculate_mi_and_bounds
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

def run_CL_for_m(device, m_size, n_size=750):
    model = SimpleNet().to(device)
    model.apply(init_weights)
    buffer = MemoryBuffer(max_size=m_size)
    
    all_test_loaders = []
    task_bounds_tracking = []
    
    # 建立全局快照字典
    supersample_memory_bank = {}
    
    for task_id in range(5):
        _, test_loader, raw_train_dataset = get_split_mnist_loaders(task_id, batch_size=128)
        all_test_loaders.append(test_loader)
        
        total_len = len(raw_train_dataset.tensors[0])
        indices = torch.randperm(total_len)[:n_size]
        trunc_data = raw_train_dataset.tensors[0][indices]
        trunc_labels = raw_train_dataset.tensors[1][indices]
        trunc_dataset = TensorDataset(trunc_data, trunc_labels)
        
        super_dataset = SupersampleDataset(trunc_dataset)
        
        # 封存状态
        supersample_memory_bank[task_id] = super_dataset
        
        new_data = super_dataset.train_data
        new_labels = super_dataset.labels.long()
        
        old_data, old_labels = buffer.get_buffer_data()
        if old_data is not None:
            combined_data = torch.cat([new_data, old_data], dim=0)
            combined_labels = torch.cat([new_labels, old_labels], dim=0)
        else:
            combined_data = new_data
            combined_labels = new_labels
            
        combined_dataset = TensorDataset(combined_data, combined_labels)
        combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
        
        _ = train_and_evaluate(model, combined_loader, test_loader, device, epochs=3)

        seen_dataset = TensorDataset(new_data, new_labels)
        buffer.update_buffer(seen_dataset, task_id)
        
    # 评估阶段
    model.eval()

    pop_loss, pop_total = 0, 0
    with torch.no_grad():
        for loader in all_test_loaders:
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = torch.max(model(imgs), 1)[1]
                pop_loss += (preds != lbls).sum().item()
                pop_total += lbls.size(0)
    pop_risk = pop_loss / pop_total
    
    emp_loss, emp_total = 0, 0
    buf_imgs, buf_lbls = buffer.get_buffer_data()
    with torch.no_grad():
        if buf_imgs is not None:
            buf_dataset = TensorDataset(buf_imgs, buf_lbls)
            buf_loader = DataLoader(buf_dataset, batch_size=128)
            for imgs, lbls in buf_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = torch.max(model(imgs), 1)[1]
                emp_loss += (preds != lbls).sum().item()
                emp_total += lbls.size(0)
        
        curr_dataset = TensorDataset(new_data, new_labels)
        curr_loader = DataLoader(curr_dataset, batch_size=128)
        for imgs, lbls in curr_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = torch.max(model(imgs), 1)[1]
            emp_loss += (preds != lbls).sum().item()
            emp_total += lbls.size(0)
            
    emp_risk = emp_loss / emp_total
    true_cl_gap = abs(pop_risk - emp_risk)
    
    final_sq, final_kl, final_wei, final_var = 0, 0, 0, 0
    
    # 旧任务信息量
    n_tilde = max(m_size // 4, 1) 

    # 在最终的 model 上，回溯测算所有 5 个任务的互信息
    for task_i in range(5):

        full_super = supersample_memory_bank[task_i]
        # 提取快照
        if task_i < 4:
            eval_size = min(n_tilde, len(full_super))
        else:
            eval_size = len(full_super)

        # 旧任务评估为 n_tilde，当前任务为 n_size
        eval_size = n_tilde if task_i < 4 else n_size

        # 固定顺序截取，不使用随机打乱
        indices = torch.arange(eval_size)
        subset_super = torch.utils.data.Subset(full_super, indices)
        super_loader = DataLoader(subset_super, batch_size=128, shuffle=False)

        # 测算该任务的理论界限 
        b = calculate_mi_and_bounds(model, super_loader, device, n=eval_size, m=m_size)

        final_sq += b['sq_bound']
        final_kl += b['bkl_bound']
        final_wei += b['wei_bound']
        final_var += b['var_bound']

    final_sq /= 5
    final_kl /= 5
    final_wei /= 5
    final_var /= 5

    return {
        'gap': true_cl_gap,
        'sq': final_sq,
        'kl': final_kl,
        'wei': final_wei,
        'var': final_var
    }

def plot_paper_fig1c(m_values, all_results):
    plt.figure(figsize=(8, 6))
    
    metrics = {
        'gap': ('Error', 'tab:blue', 'o'),
        'sq': ('Square', 'tab:orange', '+'),
        'kl': ('Binary KL', 'tab:green', '^'),
        'wei': ('Weighted', 'tab:red', '*'),
        'var': ('Variance', 'tab:purple', 'x')
    }
    
    for key, (label, color, marker) in metrics.items():
        means = np.array([np.mean(all_results[m][key]) for m in m_values])
        stds = np.array([np.std(all_results[m][key]) for m in m_values])
        
        plt.plot(m_values, means, marker=marker, label=label, color=color, linewidth=1.5, markersize=7)
        plt.fill_between(m_values, means - stds, means + stds, color=color, alpha=0.15)

    plt.title('MNIST (n = 750)', fontsize=16)
    plt.xlabel('m (Memory Buffer Size)', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.xticks(m_values)
    
    plt.yscale('linear')
    
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig('reproduced_fig1c_fix_S&01.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nFig 1(c) 复现已生成: ")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    m_values = [250, 750, 1200, 1500, 2000]
    runs_per_m = 3 
    
    results_dict = {m: {'gap':[], 'sq':[], 'kl':[], 'wei':[], 'var':[]} for m in m_values}
    
    for m in m_values:
        print(f"\n正在测算 记忆容量 m = {m} ...")
        for run_id in range(runs_per_m):
            set_seed(42 + run_id * 10) 
            print(f"运行 Trial {run_id + 1}/{runs_per_m}", end="", flush=True)
            
            res = run_CL_for_m(device, m_size=m, n_size=750)
            
            results_dict[m]['gap'].append(res['gap'])
            results_dict[m]['sq'].append(res['sq'])
            results_dict[m]['kl'].append(res['kl'])
            results_dict[m]['wei'].append(res['wei'])
            results_dict[m]['var'].append(res['var'])
            print("  (完成)")
            
    plot_paper_fig1c(m_values, results_dict)