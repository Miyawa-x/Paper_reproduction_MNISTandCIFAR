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

        # 全局 0-1 泛化误差
        print(f"正在测算 Task 0 到 Task {task_id} 的全局 0-1 泛化误差...")
        
        # Population Risk
        pop_loss, pop_total = 0, 0
        model.eval()
        with torch.no_grad():
            for i in range(task_id + 1):
                _, past_test_loader, _ = get_split_mnist_loaders(task_id=i, batch_size=128)
                for imgs, lbls in past_test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = torch.max(model(imgs), 1)[1]
                    pop_loss += (preds != lbls).sum().item()
                    pop_total += lbls.size(0)
        pop_risk = pop_loss / pop_total
        global_acc = 100.0 * (pop_total - pop_loss) / pop_total

        # Empirical Risk
        emp_loss, emp_total = 0, 0
        with torch.no_grad():
            # Buffer 里的数据
            buf_imgs, buf_lbls = buffer.get_buffer_data()
            if buf_imgs is not None:
                buf_dataset = TensorDataset(buf_imgs, buf_lbls)
                buf_loader = DataLoader(buf_dataset, batch_size=128)
                for imgs, lbls in buf_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = torch.max(model(imgs), 1)[1]
                    emp_loss += (preds != lbls).sum().item()
                    emp_total += lbls.size(0)
            
            # 当前 train_data
            curr_dataset = TensorDataset(new_data, new_labels)
            curr_loader = DataLoader(curr_dataset, batch_size=128)
            for imgs, lbls in curr_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = torch.max(model(imgs), 1)[1]
                emp_loss += (preds != lbls).sum().item()
                emp_total += lbls.size(0)
                
        emp_risk = emp_loss / emp_total if emp_total > 0 else 0
        
        # 真实loss
        true_cl_gap = abs(pop_risk - emp_risk)
        print(f"-> 全局测试误差: {pop_risk:.4f} | 全局经验误差: {emp_risk:.4f} | 真实 Gap: {true_cl_gap:.4f}")

        # 测算全局理论界限 & 越界修复
        print(f"正在测算 Task 0 到 Task {task_id} 的全局理论界限...")
        sum_mi, sum_sq, sum_bkl, sum_wei, sum_var = 0, 0, 0, 0, 0
        
        for eval_id in range(task_id + 1):
            archived_dataset = supersample_memory_bank[eval_id]
            
            if eval_id == task_id:
                # 取全量对
                eval_size = len(archived_dataset)
            else:
                target_size = max(buffer_size // (task_id + 1), 1)
                eval_size = min(target_size, len(archived_dataset))
            
            # 截取快照
            indices = torch.arange(eval_size)
            subset_super = torch.utils.data.Subset(archived_dataset, indices)
            super_loader = DataLoader(subset_super, batch_size=128, shuffle=False)
            
            b = calculate_mi_and_bounds(model, super_loader, device, n=eval_size, m=buffer_size)
            
            sum_mi += b['mi']
            sum_sq += b['sq_bound']
            sum_bkl += b['bkl_bound']
            sum_wei += b['wei_bound']
            sum_var += b['var_bound']
        
        # 算术平均
        num_tasks_seen = task_id + 1
        bounds_result = {
            'mi': sum_mi / num_tasks_seen,
            'sq_bound': sum_sq / num_tasks_seen,
            'bkl_bound': sum_bkl / num_tasks_seen,
            'wei_bound': sum_wei / num_tasks_seen,
            'var_bound': sum_var / num_tasks_seen,
            'true_01_gap': true_cl_gap,   
            'global_acc': global_acc      
        }
        
        bounds_result.update(history)
        all_tasks_history.append(bounds_result)
        print("\n")

    return all_tasks_history

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算平台: {device}")
    results = run_continual_learning(device, buffer_size=400, epochs_per_task=3)