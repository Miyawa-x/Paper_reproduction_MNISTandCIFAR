import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'MNIST_sole'))
sys.path.append(os.path.join(BASE_DIR, 'Split_MNIST'))
from Split_MNIST.split_train import run_continual_learning

def run_sole_baseline(device):
    print(f"\n{'='*20} Sole-MNIST 全局基准 {'='*20}")
    from MNIST_sole.data_setup import get_mnist_loaders
    from MNIST_sole.model import SimpleNet as SoleNet
    from MNIST_sole.basic_train import train_and_evaluate
    
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    model = SoleNet().to(device)
    history = train_and_evaluate(model, train_loader, test_loader, device, epochs=3)
    
    final_acc = history['test_acc'][-1]
    print(f"Sole-MNIST 训练结束 最终全局准确率: {final_acc:.2f}%\n")
    return final_acc

def plot_core_results(split_results, sole_final_acc, num_runs):
    tasks = range(1, 6)
    
    # 提取平均后的数据
    gaps = [abs(res['true_01_gap']) for res in split_results]
    sq_bounds = [res['sq_bound'] for res in split_results]
    bkl_bounds = [res['bkl_bound'] for res in split_results]
    wei_bounds = [res['wei_bound'] for res in split_results]
    var_bounds = [res['var_bound'] for res in split_results]
    split_accs = [res['global_acc'] for res in split_results]

    # 图 1：理论界限与经验误差
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, gaps, marker='o', label='Error', color='tab:blue', linewidth=2)
    plt.plot(tasks, sq_bounds, marker='^', label='Square',color='tab:orange', linestyle='--')
    plt.plot(tasks, bkl_bounds, marker='s', label='Binary KL',color='tab:green', linestyle='--')
    plt.plot(tasks, wei_bounds, marker='*', label='Weighted',color='tab:red', linestyle='-.')
    plt.plot(tasks, var_bounds, marker='x', label='Variance', color='tab:purple',linestyle='-.')

    plt.title(f'Generalization Bounds in Replay CL', fontsize=14)
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Generalization Error / Bounds', fontsize=12)
    plt.yscale('log') 
    plt.xticks(tasks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fig1_generalization_bounds_avg.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2：准确率演变与全局基准线
    plt.figure(figsize=(8, 5))
    plt.plot(tasks, split_accs, marker='o', label='Split-MNIST (Continual)', color='red', linewidth=2)
    plt.axhline(y=sole_final_acc, color='gray', linestyle='--', label=f'Sole-MNIST Baseline ({sole_final_acc:.1f}%)')
    
    plt.title(f'Test Accuracy', fontsize=14)
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(tasks)
    
    plt.ylim(min(min(split_accs) - 5, 80), 105) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fig2_accuracy_evolution_avg.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n实验图表已保存")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("启动泛化界限验证实验")
    
    # 跑一次 Sole 
    sole_baseline = run_sole_baseline(device)
    
    # 连续跑 10 次 Split-MNIST
    num_runs = 10
    all_runs_results = []
    
    for r in range(num_runs):
        print(f"\n正在启动第 {r+1}/{num_runs} 次独立实验")
        # 每次都会重新初始化模型和 Buffer
        results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)
        all_runs_results.append(results)

    print("\n正在计算数学期望...")
    avg_results = []
    for task_idx in range(5):
        task_dict = {}
        # 各项指标抽出来求平均
        for key in ['true_01_gap', 'sq_bound', 'bkl_bound', 'wei_bound', 'var_bound', 'global_acc']:
            task_dict[key] = np.mean([run[task_idx][key] for run in all_runs_results])
        avg_results.append(task_dict)
        
    # 绘制图表
    print("正在生成期望分析图表...")
    plot_core_results(avg_results, sole_final_acc=sole_baseline, num_runs=num_runs)