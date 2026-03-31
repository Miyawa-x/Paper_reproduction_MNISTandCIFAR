import sys
import os
import torch
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'MNIST_sole'))
sys.path.append(os.path.join(BASE_DIR, 'Split_MNIST'))
from Split_MNIST.split_train import run_continual_learning

def run_sole_baseline(device):
    print(f"\n{'='*20} 正在测算 Sole-MNIST 全局基准 {'='*20}")
    # 精确导入单体训练需要的组件
    from MNIST_sole.data_setup import get_mnist_loaders
    from MNIST_sole.model import SimpleNet as SoleNet
    from MNIST_sole.basic_train import train_and_evaluate
    
    # 加载 0-9 全量数据
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    # 实例化 10 分类模型
    model = SoleNet().to(device)
    history = train_and_evaluate(model, train_loader, test_loader, device, epochs=3)
    
    final_acc = history['test_acc'][-1]
    print(f"Sole-MNIST 训练结束 | 最终全局准确率: {final_acc:.2f}%\n")
    return final_acc

def plot_core_results(split_results, sole_final_acc):
    tasks = range(1, 6)
    
    # 提取信息论与误差数据
    gaps = [res['gap'] for res in split_results]
    sq_bounds = [res['sq_bound'] for res in split_results]
    bkl_bounds = [res['bkl_bound'] for res in split_results]
    wei_bounds = [res['wei_bound'] for res in split_results]
    var_bounds = [res['var_bound'] for res in split_results]
    
    # 提取准确率数据
    split_accs = [res['test_acc'][-1] for res in split_results]

    # 图 1：理论界限与经验误差
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, gaps, marker='o', label='Empirical Error (Gap)', color='blue', linewidth=2)
    plt.plot(tasks, sq_bounds, marker='^', label='Square-root Bound', linestyle='--')
    plt.plot(tasks, bkl_bounds, marker='s', label='Binary KL Bound', linestyle='--')
    plt.plot(tasks, wei_bounds, marker='*', label='Weighted Bound', linestyle='-.')
    plt.plot(tasks, var_bounds, marker='x', label='Variance Bound', linestyle='-.')

    plt.title('Generalization Bounds in Replay-based Continual Learning', fontsize=14)
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Generalization Error / Bounds', fontsize=12)
    plt.xticks(tasks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fig1_generalization_bounds.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2：准确率演变与全局基准线
    plt.figure(figsize=(8, 5))
    plt.plot(tasks, split_accs, marker='o', label='Split-MNIST (Continual)', color='red', linewidth=2)
    plt.axhline(y=sole_final_acc, color='gray', linestyle='--', label=f'Sole-MNIST Baseline ({sole_final_acc:.1f}%)')
    
    plt.title('Test Accuracy Evolution', fontsize=14)
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(tasks)
    
    plt.ylim(min(min(split_accs) - 5, 80), 105) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fig2_accuracy_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n实验图表已保存: \n1. fig1_generalization_bounds.png \n2. fig2_accuracy_evolution.png")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("启动 Continual Learning 泛化界限验证实验")
    
    # 全局一次性学习结果 (Sole)
    sole_baseline = run_sole_baseline(device)
    
    # 持续学习与信息论界限测算 (Split)
    print(f"\n{'='*20} 启动 Split-MNIST 训练与界限测算 {'='*20}")
    split_results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)
    
    # 绘制并保存图表
    print("\n-> 正在生成分析图表...")
    plot_core_results(split_results, sole_final_acc=sole_baseline)