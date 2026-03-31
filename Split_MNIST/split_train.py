import torch
from torch.utils.data import DataLoader, TensorDataset

from data_setup_buffer import get_split_mnist_loaders, MemoryBuffer
from basic_train import train_and_evaluate
from split_model import SimpleNet  # 输出层为2

from bounds_evaluator import SupersampleDataset, calculate_mi_and_bounds

def run_continual_learning(device, buffer_size=400, epochs_per_task=3):

    # 初始化模型 记忆
    model = SimpleNet().to(device)
    buffer = MemoryBuffer(max_size=buffer_size)

    # 泛化数据
    all_tasks_history = []

    for task_id in range(5):
        print(f"\n{'='*20} 正在开始 Task {task_id} {'='*20}")
        train_loader, test_loader, raw_train_dataset = get_split_mnist_loaders(task_id)
        old_data,old_labels = buffer.get_buffer_data()
        combined_data = None
        combined_labels = None
        
        # 新任务Tensor
        new_data = raw_train_dataset.tensors[0]
        new_labels = raw_train_dataset.tensors[1]

        # 混合新旧数据
        if old_data is not None:
            combined_data = torch.cat([new_data,old_data],dim = 0)
            combined_labels = torch.cat([new_labels,old_labels],dim = 0)
        else:
            combined_data = new_data
            combined_labels = new_labels
        
        print(f"新数据: {len(new_data)} 条", end="")
        if old_data is not None:
            print(f" | 旧数据: {len(old_data)} 条")
        else:
            print() 

        combined_dataset = TensorDataset(combined_data, combined_labels)
        combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
        
        history = train_and_evaluate(
            model, 
            combined_loader, 
            test_loader, 
            device, 
            epochs=epochs_per_task
        )
        
        # 算出实测的 Gap
        final_train_loss = history['train_loss'][-1]
        final_test_loss = history['test_loss'][-1]
        generalization_gap = abs(final_test_loss - final_train_loss)
        print(f"Task {task_id} 结束 | Train Loss: {final_train_loss:.4f} | Test Loss: {final_test_loss:.4f} | 泛化差距 (Gap): {generalization_gap:.4f}")

        print(f"--- 正在测算 Task {task_id} 的信息论泛化界限 ---")
        super_dataset = SupersampleDataset(raw_train_dataset)
        super_loader = DataLoader(super_dataset, batch_size=128, shuffle=False)
        n_samples = len(raw_train_dataset)

        bounds_result = calculate_mi_and_bounds(model, super_loader, device, n=n_samples, m=buffer_size)
        bounds_result['gap'] = generalization_gap
        
        # 存入历史记录
        all_tasks_history.append(bounds_result)
        print("\n")

    return all_tasks_history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算平台: {device}")
    results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)