import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
import random

# ==========================================
# 0. 全局设置与随机种子
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. 配置与超参数
# ==========================================
CONFIG = {
    'hidden_dim': 32,         
    'lr': 0.005,
    'epochs': 1000,
    'batch_size': 16,
    'split_ratio': [0.3, 0.1, 0.6], # Train / Val / Test
    'device': torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    'print_freq': 1
}

# ==========================================
# 2. 数据集处理 (BioreactorDataset)
# ==========================================
class BioreactorDataset:
    def __init__(self, csv_path, split_ratio=[0.8, 0.1, 0.1]):
        self.df = pd.read_csv(csv_path)
        
        # 字段定义
        self.state_cols = ['VCD', 'Glc', 'Gln', 'Amm', 'Lac', 'product']
        self.input_cols = ['Cmd_Temp', 'Cmd_pH', 'Cmd_Stirring']
        self.vol_col = 'Volume'
        self.feed_glc_mass_col = 'Cmd_Feed_Glc_Mass' 
        self.feed_gln_mass_col = 'Cmd_Feed_Gln_Mass'
        self.feed_vol_col = 'Cmd_Feed_Vol'
        
        # --- 数据集分割逻辑 ---
        self.all_exp_ids = self.df['Exp_ID'].unique()
        n_total = len(self.all_exp_ids)
        
        shuffled_ids = np.random.permutation(self.all_exp_ids)
        
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        self.train_ids = shuffled_ids[:n_train]
        self.val_ids = shuffled_ids[n_train:n_train+n_val]
        self.test_ids = shuffled_ids[n_train+n_val:]
        
        if len(self.test_ids) == 0 and n_total > 1:
            self.test_ids = shuffled_ids[-1:]
            self.train_ids = shuffled_ids[:-1]
            self.val_ids = []
            print("Warning: Dataset too small for 8:1:1 split. Fallback to Train/Test only.")

        print(f"Dataset Split -> Train: {len(self.train_ids)}, Val: {len(self.val_ids)}, Test: {len(self.test_ids)}")

        # --- 归一化统计量 (仅基于训练集!) ---
        train_df = self.df[self.df['Exp_ID'].isin(self.train_ids)]
        
        self.state_mean = torch.tensor(train_df[self.state_cols].mean().values, dtype=torch.float32)
        self.state_std = torch.tensor(train_df[self.state_cols].std().values + 1e-6, dtype=torch.float32)
        
        self.input_mean = torch.tensor(train_df[self.input_cols].mean().values, dtype=torch.float32)
        self.input_std = torch.tensor(train_df[self.input_cols].std().values + 1e-6, dtype=torch.float32)

    def get_exp_data(self, exp_id):
        exp_df = self.df[self.df['Exp_ID'] == exp_id].sort_values('time[h]')
        t_np = exp_df['time[h]'].values.astype(np.float32)
        
        targets = torch.tensor(exp_df[self.state_cols].values, dtype=torch.float32)
        y0 = targets[0]
        
        input_funcs = []
        for col in self.input_cols:
            f = interp1d(t_np, exp_df[col].values, kind='previous', fill_value="extrapolate")
            input_funcs.append(f)
            
        vol_spline = CubicSpline(t_np, exp_df[self.vol_col].values)
        
        intervals = np.diff(t_np)
        if len(intervals) > 0:
            intervals = np.append(intervals, intervals[-1])
        else:
            intervals = np.array([1.0])

        glc_rate = exp_df[self.feed_glc_mass_col].values / (intervals + 1e-6)
        gln_rate = exp_df[self.feed_gln_mass_col].values / (intervals + 1e-6)
        vol_flow_rate = exp_df[self.feed_vol_col].values / (intervals + 1e-6)
        
        glc_rate_func = interp1d(t_np, glc_rate, kind='previous', fill_value="extrapolate")
        gln_rate_func = interp1d(t_np, gln_rate, kind='previous', fill_value="extrapolate")
        flow_rate_func = interp1d(t_np, vol_flow_rate, kind='previous', fill_value="extrapolate")
        
        return {
            'times': torch.tensor(t_np),
            'y0': y0,
            'targets': targets,
            'input_funcs': input_funcs,
            'vol_spline': vol_spline,
            'glc_rate_func': glc_rate_func,
            'gln_rate_func': gln_rate_func,
            'flow_rate_func': flow_rate_func,
            'exp_id': exp_id
        }

# ==========================================
# 3. 混合模型定义 (Hybrid ODE)
# ==========================================
class KineticsNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class HybridODEFunc(nn.Module):
    def __init__(self, kinetics_net, exp_data_dict, stats):
        super().__init__()
        self.net = kinetics_net
        self.data = exp_data_dict
        
        self.register_buffer('state_mean', stats['state_mean'])
        self.register_buffer('state_std', stats['state_std'])
        self.register_buffer('input_mean', stats['input_mean'])
        self.register_buffer('input_std', stats['input_std'])
        
    def forward(self, t, y):
        t_np = t.item()
        
        # --- A. Inputs & Normalization ---
        controls_np = np.array([f(t_np) for f in self.data['input_funcs']], dtype=np.float32)
        controls = torch.from_numpy(controls_np).to(y.device)
        
        norm_y = (y - self.state_mean) / self.state_std
        norm_controls = (controls - self.input_mean) / self.input_std
        nn_input = torch.cat([norm_y, norm_controls], dim=-1)
        
        # --- B. Kinetics Prediction ---
        rates = self.net(nn_input)
        
        # --- C. Physics (Volume, Feed, Dilution) ---
        V = torch.tensor(self.data['vol_spline'](t_np), dtype=torch.float32).to(y.device)
        F_in = torch.tensor(self.data['flow_rate_func'](t_np), dtype=torch.float32).to(y.device)
        D = F_in / V # Dilution Rate
        
        glc_in = torch.tensor(self.data['glc_rate_func'](t_np), dtype=torch.float32).to(y.device)
        gln_in = torch.tensor(self.data['gln_rate_func'](t_np), dtype=torch.float32).to(y.device)

        # --- D. Mass Balance ---
        X_v, Glc, Gln, Amm, Lac, Prod = y[0], y[1], y[2], y[3], y[4], y[5]
        X_v_safe = torch.relu(X_v) # Physics constraint
        
        mu, q_glc, q_gln, q_amm, q_lac, q_prod = rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]

        d_vcd = (mu - D) * X_v_safe
        d_glc = -q_glc * X_v_safe - D * Glc + (glc_in / V)
        d_gln = -q_gln * X_v_safe - D * Gln + (gln_in / V)
        d_amm = q_amm * X_v_safe - D * Amm
        d_lac = q_lac * X_v_safe - D * Lac
        d_prod = q_prod * X_v_safe - D * Prod
        
        return torch.stack([d_vcd, d_glc, d_gln, d_amm, d_lac, d_prod])

# ==========================================
# 4. 训练、验证与测试
# ==========================================
def calculate_nrmse_std(pred, target, std_val):
    mse = torch.mean((pred - target) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    nrmse = rmse / (std_val + 1e-6)
    return nrmse.mean().item() * 100

def run_evaluation(net, dataset, stats, target_ids, dataset_name="Val"):
    if len(target_ids) == 0:
        return 0.0
        
    net.eval()
    total_nrmse = 0
    count = 0
    
    with torch.no_grad():
        for exp_id in target_ids:
            exp_data = dataset.get_exp_data(exp_id)
            ode_func = HybridODEFunc(net, exp_data, stats).to(CONFIG['device'])
            
            y0 = exp_data['y0'].to(CONFIG['device'])
            t = exp_data['times'].to(CONFIG['device'])
            targets = exp_data['targets'].to(CONFIG['device'])
            
            try:
                pred_y = odeint(ode_func, y0, t, method='rk4')
                nrmse = calculate_nrmse_std(pred_y, targets, dataset.state_std.to(CONFIG['device']))
                total_nrmse += nrmse
                count += 1
            except Exception as e:
                print(f"Error evaluating {exp_id}: {e}")
            
    avg_nrmse = total_nrmse / max(count, 1)
    return avg_nrmse

def train_pipeline():
    # 1. 初始化
    dataset = BioreactorDataset('data/test.csv', split_ratio=CONFIG['split_ratio'])
    stats = {
        'state_mean': dataset.state_mean.to(CONFIG['device']),
        'state_std': dataset.state_std.to(CONFIG['device']),
        'input_mean': dataset.input_mean.to(CONFIG['device']),
        'input_std': dataset.input_std.to(CONFIG['device'])
    }
    
    net = KineticsNet(input_dim=9, hidden_dim=CONFIG['hidden_dim'], output_dim=6).to(CONFIG['device'])
    optimizer = optim.Adam(net.parameters(), lr=CONFIG['lr'])
    
    # 2. 训练循环
    print(f"\n--- Starting Training (Epochs: {CONFIG['epochs']}) ---")
    best_val_nrmse = float('inf')
    best_model_state = None
    
    for epoch in range(CONFIG['epochs']):
        net.train()
        batch_loss = 0
        epoch_train_ids = np.random.permutation(dataset.train_ids)
        
        for exp_id in epoch_train_ids:
            exp_data = dataset.get_exp_data(exp_id)
            ode_func = HybridODEFunc(net, exp_data, stats).to(CONFIG['device'])
            
            y0 = exp_data['y0'].to(CONFIG['device'])
            t = exp_data['times'].to(CONFIG['device'])
            targets = exp_data['targets'].to(CONFIG['device'])
            
            pred_y = odeint(ode_func, y0, t, method='rk4')
            
            # Loss: Standardized MSE
            norm_pred = (pred_y - stats['state_mean']) / stats['state_std']
            norm_target = (targets - stats['state_mean']) / stats['state_std']
            loss = torch.mean((norm_pred - norm_target)**2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            batch_loss += loss.item()
            
        avg_train_loss = batch_loss / len(epoch_train_ids)
        
        if (epoch + 1) % CONFIG['print_freq'] == 0:
            val_nrmse = run_evaluation(net, dataset, stats, dataset.val_ids, "Val")
            print(f"Epoch {epoch+1:04d} | Train Loss: {avg_train_loss:.5f} | Val NRMSE: {val_nrmse:.2f}%")
            
            if val_nrmse < best_val_nrmse and len(dataset.val_ids) > 0:
                best_val_nrmse = val_nrmse
                best_model_state = net.state_dict()
                
    if best_model_state is not None:
        print(f"\nRestoring best model with Val NRMSE: {best_val_nrmse:.2f}%")
        net.load_state_dict(best_model_state)
            
    return net, dataset, stats

# ==========================================
# 5. 可视化函数 (新增)
# ==========================================
def visualize_prediction(net, dataset, stats, exp_id, save_path=None):
    """
    绘制预测轨迹 vs 真实值
    - 预测值: 连续线 (Dense time points)
    - 真实值: 散点 (Sparse experimental points)
    """
    net.eval()
    exp_data = dataset.get_exp_data(exp_id)
    ode_func = HybridODEFunc(net, exp_data, stats).to(CONFIG['device'])
    
    # 1. 真实值 (Ground Truth)
    t_true = exp_data['times'].cpu().numpy()
    y_true = exp_data['targets'].cpu().numpy()
    
    # 2. 预测值 (Prediction on Dense Grid)
    # 创建更密的时间网格用于画平滑曲线 (例如 100 个点)
    t_dense = torch.linspace(t_true[0], t_true[-1], 100).to(CONFIG['device'])
    y0 = exp_data['y0'].to(CONFIG['device'])
    
    with torch.no_grad():
        # 积分预测
        pred_y = odeint(ode_func, y0, t_dense, method='rk4')
        
    t_pred = t_dense.cpu().numpy()
    y_pred = pred_y.cpu().numpy()
    
    # 3. 绘图
    cols = dataset.state_cols
    # 创建 2x3 网格
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        ax = axes[i]
        
        # 画线 (预测)
        ax.plot(t_pred, y_pred[:, i], label='HR Model Pred', color='tab:blue', linewidth=2)
        
        # 画点 (真实)
        ax.scatter(t_true, y_true[:, i], label='Exp Data', color='tab:red', s=40, zorder=5)
        
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Concentration')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 只在第一个图显示图例，避免遮挡
        if i == 0:
            ax.legend(loc='best')
            
    plt.suptitle(f'Hybrid Model Prediction: {exp_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == '__main__':
    if os.path.exists('data/test.csv'):
        # 1. 训练流程
        model, ds, stats = train_pipeline()
        
        # 2. 测试集评估
        print("\n--- Final Evaluation on Test Set ---")
        test_nrmse = run_evaluation(model, ds, stats, ds.test_ids, "Test")
        print(f"Final Test Set NRMSE: {test_nrmse:.2f}%")
        
        # 3. 对测试集的样本进行预测并绘图
        if len(ds.test_ids) > 0:
            # 随机选一个测试样本或者遍历所有
            sample_ids = ds.test_ids[:3] # 取前3个测试样本看看
            
            for sid in sample_ids:
                print(f"\nVisualizing prediction for test sample: {sid}")
                
                # A. 绘图并保存
                visualize_prediction(model, ds, stats, sid, save_path=f'plot_{sid}.png')
                
                # B. 保存 CSV 数据 (可选)
                # ... (同之前的导出逻辑，此处略以保持简洁) ...
            
    else:
        print("Error: 'novel.csv' not found.")