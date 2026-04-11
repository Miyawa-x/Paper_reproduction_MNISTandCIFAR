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

    all_tasks_history = []
    
    # 全局缓存字典
    supersample_memory_bank = {}

    for task_id in range(5):
        print(f"\n正在开始 Task {task_id}")
        train_loader, test_loader, raw_train_dataset = get_split_mnist_loaders(task_id)
        super_dataset = SupersampleDataset(raw_train_dataset)
        
        # 状态存储
        supersample_memory_bank[task_id] = super_dataset
        
        # 模型只拿超样本中分配为 Train 的数据去训练
        new_data = super_dataset.train_data
        new_labels = super_dataset.labels.long()

        old_data, old_labels = buffer.get_buffer_data()
        
        # 混合数据
        if old_data is not None:
            combined_data = torch.cat([new_data, old_data], dim=0)
            combined_labels = torch.cat([new_labels, old_labels], dim=0)
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
        
        seen_dataset = TensorDataset(new_data, new_labels)
        buffer.update_buffer(seen_dataset, task_id)

        # 算出实测的 Gap
        final_train_loss = history['train_loss'][-1]
        final_test_loss = history['test_loss'][-1]
        generalization_gap = abs(final_test_loss - final_train_loss)
        print(f"Task {task_id} 结束 | Train Loss: {final_train_loss:.4f} | Test Loss: {final_test_loss:.4f} | 泛化差距 (Gap): {generalization_gap:.4f}")

        # 截止到当前的全局平均理论界限
        print(f"正在测算 Task 0 到 Task {task_id} 的全局理论界限")
        
        sum_mi, sum_sq, sum_bkl, sum_wei, sum_var = 0, 0, 0, 0, 0
        
        model.eval()
        # 遍历任务重新在当前model上提取 Delta
        for eval_id in range(task_id + 1):
            archived_dataset = supersample_memory_bank[eval_id]
            
            # 最新任务的分母大，旧任务压进 Buffer 后分母小
            if eval_id == task_id:
                eval_size = len(new_data)
            else:
                eval_size = max(buffer_size // (task_id + 1), 1)
            
            # 读取配对好的数据
            indices = torch.arange(eval_size)
            subset_super = torch.utils.data.Subset(archived_dataset, indices)
            super_loader = DataLoader(subset_super, batch_size=128, shuffle=False)
            
            b = calculate_mi_and_bounds(model, super_loader, device, n=eval_size, m=buffer_size)
            
            sum_mi += b['mi']
            sum_sq += b['sq_bound']
            sum_bkl += b['bkl_bound']
            sum_wei += b['wei_bound']
            sum_var += b['var_bound']
        
        # 除以 T 求算术平均
        num_tasks_seen = task_id + 1
        bounds_result = {
            'mi': sum_mi / num_tasks_seen,
            'sq_bound': sum_sq / num_tasks_seen,
            'bkl_bound': sum_bkl / num_tasks_seen,
            'wei_bound': sum_wei / num_tasks_seen,
            'var_bound': sum_var / num_tasks_seen,
        }
        bounds_result['gap'] = generalization_gap
        bounds_result.update(history)
        
        print(f"测算 Task 0 到 Task {task_id} 的平均准确率")
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(task_id + 1):
                _, past_test_loader, _ = get_split_mnist_loaders(task_id=i, batch_size=128)
                
                for images, labels in past_test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    
        global_acc = 100.0 * total_correct / total_samples
        print(f"全局平均准确率: {global_acc:.2f}%\n")
        
        bounds_result['global_acc'] = global_acc
        bounds_result['true_01_gap'] = generalization_gap

        all_tasks_history.append(bounds_result)
        print("\n")

    return all_tasks_history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算平台: {device}")
    results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)