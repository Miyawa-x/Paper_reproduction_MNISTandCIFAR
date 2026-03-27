import torch
from torch.utils.data import DataLoader, TensorDataset

from data_setup_buffer import get_split_mnist_loaders, MemoryBuffer
from basic_train import train_and_evaluate
from split_model import SimpleNet  # 输出层为2

def run_continual_learning(device, buffer_size=400, epochs_per_task=3):

    # 初始化模型 记忆
    model = SimpleNet().to(device)
    buffer = MemoryBuffer(max_size=buffer_size)

    # 用来记录所有任务跑完后的泛化数据（为复现论文图表做准备）
    all_tasks_history = []

    for task_id in range(5):
        print(f"\n{'='*20} 正在开始 Task {task_id} {'='*20}")
        
        # TODO 3.1: 调用 data_setup 里的函数，获取当前任务的新数据
        # train_loader, test_loader, raw_train_dataset = ...
        
        # TODO 3.2: 从 Buffer 获取旧记忆
        # old_data, old_labels = ...
        
        # 准备用来训练的混合数据
        combined_data = None
        combined_labels = None
        
        # 提取当前新任务的 Tensor
        # Hint: raw_train_dataset.tensors[0] 是数据，tensors[1] 是标签
        new_data = raw_train_dataset.tensors[0]
        new_labels = raw_train_dataset.tensors[1]
        
        # TODO 3.3: 新旧数据混合 (核心难点)
        # 如果 old_data 存在（即不是第一个任务）：
        #   使用 torch.cat 将 new_data 和 old_data 拼接 (注意维度 dim=0)
        #   将 new_labels 和 old_labels 拼接
        # 如果 old_data 不存在：
        #   combined_data 和 labels 直接等于 new_data 和 labels
        
        print(f"[数据混合] 当前任务新数据: {len(new_data)} 条", end="")
        if old_data is not None:
            print(f" | 回放旧数据: {len(old_data)} 条")
        else:
            print() # 换行
            
        # TODO 3.4: 将混合后的 Tensor 重新打包成 DataLoader
        # Hint: 先套一层 TensorDataset，再传给 DataLoader (一定要设置 shuffle=True 彻底打乱新旧数据！)
        # combined_loader = ...
        
        # TODO 3.5: 调用你的训练引擎开始训练
        # 把 combined_loader (混合训练集) 和 test_loader (当前任务测试集) 传进 train_and_evaluate
        # history = ...
        
        # TODO 3.6: 当前任务学完了，把它的原始数据丢进 Buffer 里为未来做准备
        # buffer.update_buffer(...)
        
        # 保存这一个 Task 的训练日志
        all_tasks_history.append(history)
        
        # --- 论文核心数据提取 ---
        # 提取最后一个 Epoch 的 Train Loss 和 Test Loss
        final_train_loss = history['train_loss'][-1]
        final_test_loss = history['test_loss'][-1]
        generalization_gap = abs(final_test_loss - final_train_loss)
        print(f"-> Task {task_id} 结束 | Train Loss: {final_train_loss:.4f} | Test Loss: {final_test_loss:.4f} | 泛化差距 (Gap): {generalization_gap:.4f}")

    return all_tasks_history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算平台: {device}")
    
    # 模拟论文中 M=400 的设定
    results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)