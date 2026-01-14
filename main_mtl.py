import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
import random
import copy

# ==========================================
# 0. Global Settings & Random Seed
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
CONFIG = {
    "hidden_dim": 64,             # 稍微增加隐层宽度以适应多任务
    "embedding_dim": 2,           # [New] 产品嵌入维度 (2D 便于可视化)
    "target_product": "HP5",      # [New] 目标新产品 ID
    "n_target_shots": 2,          # [New] 训练集中包含多少个目标产品的 Run (Few-shot)
    
    "lr": 0.002,
    "epochs": 200,
    "patience": 40,
    "batch_size": 4, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    "print_freq": 1,
    "ode_method": "rk4",       # 使用自适应步长更稳
    "ode_rtol": 1e-5,
    "ode_atol": 1e-7,
    "loss_type": "relative_mse",
}

print(f"Using device: {CONFIG['device']}")

# ==========================================
# 2. Dataset Handler (Multi-Task Logic)
# ==========================================
class BioreactorDataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.state_cols = ["VCD", "Glc", "Gln", "Amm", "Lac", "product"]
        self.input_cols = ["Cmd_Temp", "Cmd_pH", "Cmd_Stirring"]
        self.vol_col = "Volume"
        self.feed_glc_mass_col = "Cmd_Feed_Glc_Mass"
        self.feed_gln_mass_col = "Cmd_Feed_Gln_Mass"
        self.feed_vol_col = "Cmd_Feed_Vol"
        self.prod_id_col = "Product_ID"

        # --- [关键修改 1] 构建全局唯一 ID (Unique_ID) ---
        # 格式: "HP1_ExpHist00"
        self.df["Unique_ID"] = self.df[self.prod_id_col].astype(str) + "_" + self.df["Exp_ID"].astype(str)
        
        # --- Product Encoding ---
        unique_prods = sorted(self.df[self.prod_id_col].unique())
        self.prod_map = {name: i for i, name in enumerate(unique_prods)}
        self.num_products = len(unique_prods)
        print(f"Found Products: {self.prod_map}")

        # --- Custom Split Strategy ---
        # [关键修改 2] 使用 Unique_ID 进行唯一性列表获取
        self.all_exp_ids = self.df["Unique_ID"].unique()
        
        # 获取映射关系：Unique_ID -> Product_ID
        # 例如: 'HP1_ExpHist00' -> 'HP1'
        exp_to_prod = self.df.groupby("Unique_ID")[self.prod_id_col].first().to_dict()
        
        historical_exps = []
        target_exps = []
        
        target_name = CONFIG["target_product"]
        
        for uid in self.all_exp_ids:
            p_name = exp_to_prod[uid]
            if p_name == target_name:
                target_exps.append(uid)
            else:
                historical_exps.append(uid)
        
        # ... (后续划分逻辑保持不变，因为 uid 已经是唯一的字符串了) ...
        random.shuffle(target_exps)
        
        train_target = target_exps[:CONFIG["n_target_shots"]]
        self.train_ids = np.concatenate([historical_exps, train_target])
        self.test_ids = target_exps[CONFIG["n_target_shots"]:]
        
        np.random.shuffle(self.train_ids)
        n_val = int(len(self.train_ids) * 0.1)
        if n_val < 1: n_val = 1
        
        self.val_ids = self.train_ids[:n_val]
        self.train_ids = self.train_ids[n_val:]

        print(f"Split Strategy for Target '{target_name}':")
        print(f"  Train: {len(self.train_ids)} runs")
        print(f"  Val  : {len(self.val_ids)} runs")
        print(f"  Test : {len(self.test_ids)} runs")

        # --- Global Statistics (Only from Train set) ---
        # [关键修改 3] 使用 Unique_ID 过滤数据
        train_df = self.df[self.df["Unique_ID"].isin(self.train_ids)]
        
        self.state_mean = torch.tensor(train_df[self.state_cols].mean().values, dtype=torch.float32)
        self.state_std = torch.tensor(train_df[self.state_cols].std().values + 1e-6, dtype=torch.float32)
        self.state_mean = torch.nan_to_num(self.state_mean, nan=0.0)
        self.state_std = torch.nan_to_num(self.state_std, nan=1.0)
        
        self.input_mean = torch.tensor(train_df[self.input_cols].mean().values, dtype=torch.float32)
        self.input_std = torch.tensor(train_df[self.input_cols].std().values + 1e-6, dtype=torch.float32)

    def get_exp_events(self, unique_id):
        # [关键修改 4] 使用 Unique_ID 过滤单次实验数据
        exp_df = self.df[self.df["Unique_ID"] == unique_id].sort_values("time[h]")
        t_np = exp_df["time[h]"].values.astype(np.float32)
        
        # Get Product ID
        p_name = exp_df[self.prod_id_col].iloc[0]
        p_idx = self.prod_map[p_name]

        targets_np = exp_df[self.state_cols].values.astype(np.float32)
        mask_np = ~np.isnan(targets_np)
        targets = torch.tensor(np.nan_to_num(targets_np, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32)

        y0 = targets[0].clone()
        if torch.isnan(y0).any():
            for i in range(len(y0)):
                if torch.isnan(y0[i]):
                    valid = targets[:, i][~torch.isnan(targets[:, i])]
                    y0[i] = valid[0] if len(valid) > 0 else 0.0

        input_funcs = []
        for col in self.input_cols:
            vals = exp_df[col].interpolate(method="linear").bfill().ffill().values
            if np.isnan(vals).any(): vals = np.nan_to_num(vals, 0.0)
            f = interp1d(t_np, vals, kind="previous", fill_value="extrapolate")
            input_funcs.append(f)

        events = []
        for _, row in exp_df.iterrows():
            t = float(row["time[h]"])
            glc_m = float(row[self.feed_glc_mass_col]) if not pd.isna(row[self.feed_glc_mass_col]) else 0.0
            gln_m = float(row[self.feed_gln_mass_col]) if not pd.isna(row[self.feed_gln_mass_col]) else 0.0
            f_vol = float(row[self.feed_vol_col]) if not pd.isna(row[self.feed_vol_col]) else 0.0

            if glc_m > 1e-6 or gln_m > 1e-6 or f_vol > 1e-6:
                events.append({"time": t, "glc_mass": glc_m, "gln_mass": gln_m, "vol_add": f_vol})

        events.sort(key=lambda x: x["time"])
        initial_vol = float(exp_df.iloc[0][self.vol_col])

        return {
            "times": torch.tensor(t_np),
            "y0": y0,
            "targets": targets,
            "mask": mask,
            "input_funcs": input_funcs,
            "events": events,
            "initial_vol": initial_vol,
            "exp_id": unique_id, # 返回新的 ID 以便打印
            "product_idx": torch.tensor(p_idx, dtype=torch.long)
        }

# ==========================================
# 3. Hybrid Model with Embeddings [Updated]
# ==========================================
class KineticsNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_products, embedding_dim):
        super().__init__()
        # [New] Embedding Layer
        self.embedding = nn.Embedding(num_products, embedding_dim)
        
        # Total input = State(6) + Control(3) + Embedding(2) = 11
        total_input_dim = input_dim + embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # Init embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        last_layer = self.net[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=1e-4)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x, product_idx):
        """
        x: [Batch, 9] (State + Control)
        product_idx: [Batch] or Scalar (Int)
        """
        # 1. Get embedding for the product
        emb = self.embedding(product_idx) # [Batch, emb_dim]
        
        # 2. Concatenate with physical inputs
        # Handle broadcasting if x is [T, D] and emb is [1, D]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).expand(x.size(0), -1)
        elif emb.size(0) != x.size(0):
            emb = emb.expand(x.size(0), -1)
            
        combined_input = torch.cat([x, emb], dim=-1)
        
        return self.net(combined_input)

# ==========================================
# 4. ODE Func (Context Aware)
# ==========================================
class ReactionODEFunc(nn.Module):
    def __init__(self, kinetics_net, input_funcs, stats, product_idx):
        super().__init__()
        self.net = kinetics_net
        self.input_funcs = input_funcs
        self.product_idx = product_idx # Store current product ID
        
        self.register_buffer("state_mean", stats["state_mean"])
        self.register_buffer("state_std", stats["state_std"])
        self.register_buffer("input_mean", stats["input_mean"])
        self.register_buffer("input_std", stats["input_std"])

    def forward(self, t, y):
        t_np = t.item()
        controls_np = np.array([f(t_np) for f in self.input_funcs], dtype=np.float32)
        controls = torch.from_numpy(controls_np).to(y.device)

        norm_y = (y - self.state_mean) / self.state_std
        norm_controls = (controls - self.input_mean) / self.input_std
        
        # Combine physical inputs
        nn_input = torch.cat([norm_y, norm_controls], dim=-1) # [9]

        # [Modified] Pass product ID to net
        rates = self.net(nn_input.unsqueeze(0), self.product_idx).squeeze(0)
        
        # Rate Clamping for stability
        rates = torch.clamp(rates, min=-20.0, max=20.0)
        
        mu, q_glc, q_gln, q_amm, q_lac, q_prod = rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]

        X_v = y[0]
        X_v_safe = torch.nn.functional.softplus(X_v)
        q_prod = torch.relu(q_prod)

        d_vcd = mu * X_v_safe
        d_glc = -q_glc * X_v_safe
        d_gln = -q_gln * X_v_safe
        d_amm = q_amm * X_v_safe
        d_lac = q_lac * X_v_safe
        d_prod = q_prod * X_v_safe
        return torch.stack([d_vcd, d_glc, d_gln, d_amm, d_lac, d_prod])

# ==========================================
# 5. Solver (Updated for Prod ID)
# ==========================================
def solve_with_events(net, exp_data, stats, t_eval_points=None):
    device = CONFIG["device"]
    y_current = exp_data["y0"].to(device)
    current_vol = exp_data["initial_vol"]
    product_idx = exp_data["product_idx"].to(device) # Get ID

    if t_eval_points is None:
        t_targets = exp_data["times"].cpu().numpy()
    else:
        t_targets = t_eval_points.cpu().numpy()

    events = exp_data["events"]
    feed_times = [e["time"] for e in events]
    all_stops = sorted(list(set(np.concatenate([t_targets, feed_times]))))

    pred_dict = {}
    if abs(all_stops[0] - t_targets[0]) < 1e-4:
        pred_dict[all_stops[0]] = y_current.clone()

    # [Modified] Pass product_idx to ODE Func
    ode_func = ReactionODEFunc(net, exp_data["input_funcs"], stats, product_idx).to(device)
    current_t = all_stops[0]

    for next_t in all_stops[1:]:
        if next_t <= current_t: continue
        t_span = torch.tensor([current_t, next_t], dtype=torch.float32).to(device)
        
        # Adaptive solver for stability
        sol = odeint(ode_func, y_current, t_span, 
                     method=CONFIG["ode_method"], 
                     rtol=CONFIG["ode_rtol"], atol=CONFIG["ode_atol"])
        y_next = sol[-1]

        events_now = [e for e in events if abs(e["time"] - next_t) < 1e-4]
        if len(events_now) > 0:
            vcd, glc, gln, amm, lac, prod = y_next
            total_vcd = vcd * current_vol
            total_glc = glc * current_vol
            total_gln = gln * current_vol
            total_amm = amm * current_vol
            total_lac = lac * current_vol
            total_prod = prod * current_vol
            for e in events_now:
                total_glc += e["glc_mass"]
                total_gln += e["gln_mass"]
                current_vol += e["vol_add"]
            y_next = torch.stack([
                total_vcd / current_vol, total_glc / current_vol, total_gln / current_vol,
                total_amm / current_vol, total_lac / current_vol, total_prod / current_vol
            ])

        if np.any(np.isclose(t_targets, next_t, atol=1e-4)):
            pred_dict[next_t] = y_next.clone()
        current_t = next_t
        y_current = y_next

    final_preds = []
    for t in t_targets:
        closest_t = min(pred_dict.keys(), key=lambda x: abs(x - t))
        final_preds.append(pred_dict[closest_t])
    return torch.stack(final_preds)

# ==========================================
# 6. Loss & Helpers
# ==========================================
class EarlyStopping:
    def __init__(self, patience=20, delta=0, save_path="best_mtl_model.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, stats=None):
        if np.isnan(val_loss): return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, stats)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, stats)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, stats):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stats': stats
        }
        torch.save(checkpoint, self.save_path)

def masked_relative_mse(pred, target, mask, std_vec):
    eps = 1e-6
    diff = (pred - target) * mask
    n_eff = torch.clamp(mask.sum(dim=0), min=1.0)
    mse = (diff**2).sum(dim=0) / n_eff
    rmse = torch.sqrt(mse)
    std = torch.clamp(std_vec, min=eps)
    return rmse / std

# ==========================================
# 7. Training Pipeline
# ==========================================
def run_evaluation(net, dataset, stats, target_ids):
    if len(target_ids) == 0: return 0.0, np.zeros(6)
    net.eval()
    total_nrmse_vec = torch.zeros(6).to(CONFIG["device"])
    count = 0

    with torch.no_grad():
        for exp_id in target_ids:
            exp_data = dataset.get_exp_events(exp_id)
            targets = exp_data["targets"].to(CONFIG["device"])
            mask = exp_data["mask"].to(CONFIG["device"])
            try:
                pred_y = solve_with_events(net, exp_data, stats)
                if torch.isnan(pred_y).any(): continue
                
                nrmse_vec = masked_relative_mse(pred_y, targets, mask, stats["state_std"])
                total_nrmse_vec += nrmse_vec
                count += 1
            except Exception as e:
                print(f"Error evaluating {exp_id}: {e}")

    avg_nrmse_vec = total_nrmse_vec / max(count, 1)
    avg_scalar = avg_nrmse_vec.mean().item()
    return avg_scalar, avg_nrmse_vec.cpu().numpy()

def train_pipeline(file_path):
    dataset = BioreactorDataset(file_path)
    stats = {
        "state_mean": dataset.state_mean.to(CONFIG["device"]),
        "state_std": dataset.state_std.to(CONFIG["device"]),
        "input_mean": dataset.input_mean.to(CONFIG["device"]),
        "input_std": dataset.input_std.to(CONFIG["device"]),
    }

    # [Modified] Initialize Net with Embedding params
    net = KineticsNet(input_dim=9, output_dim=6, 
                      num_products=dataset.num_products, 
                      embedding_dim=CONFIG["embedding_dim"]).to(CONFIG["device"])

    optimizer = optim.Adam(net.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
    early_stopping = EarlyStopping(patience=CONFIG["patience"], save_path="best_mtl_model.pth")

    print(f"\n--- Starting Training (Max Epochs: {CONFIG['epochs']}) ---")
    
    accumulation_steps = CONFIG["batch_size"]

    for epoch in range(CONFIG["epochs"]):
        net.train()
        batch_loss = 0
        optimizer.zero_grad()
        epoch_train_ids = np.random.permutation(dataset.train_ids)

        for i, exp_id in enumerate(epoch_train_ids):
            exp_data = dataset.get_exp_events(exp_id)
            targets = exp_data["targets"].to(CONFIG["device"])
            mask = exp_data["mask"].to(CONFIG["device"])

            # Forward
            try:
                pred_y = solve_with_events(net, exp_data, stats)
            except Exception: continue

            # Loss
            if CONFIG["loss_type"] == "mse":
                # Implement standard mse if needed
                pass 
            else:
                nrmse_vec = masked_relative_mse(pred_y, targets, mask, stats["state_std"])
                loss = torch.mean(nrmse_vec)

            if torch.isnan(loss): continue

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_train_ids):
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

            batch_loss += loss.item() * accumulation_steps

        avg_loss = batch_loss / len(epoch_train_ids)

        # Validation
        if (epoch + 1) % CONFIG["print_freq"] == 0:
            val_avg, val_vec = run_evaluation(net, dataset, stats, dataset.val_ids)
            print(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.5f} | Val Avg: {val_avg:.2f} | Prod Err: {val_vec[5]:.2f}")
            
            scheduler.step(val_avg)
            early_stopping(val_avg, net, stats=stats)
            if early_stopping.early_stop:
                print("--- Early stopping triggered ---")
                break

    print(f"\nLoading best model...")
    checkpoint = torch.load(early_stopping.save_path, map_location=CONFIG["device"])
    net.load_state_dict(checkpoint['model_state_dict'])
    return net, dataset, stats

# ==========================================
# 8. Visualization (Embedding & Trajectories)
# ==========================================
def plot_product_embeddings(net, dataset, save_path="embedding_space.png"):
    """
    可视化不同产品的嵌入向量，展示它们在隐空间中的关系。
    """
    net.eval()
    if CONFIG["embedding_dim"] != 2:
        print("Warning: Only 2D embeddings are currently supported for direct plotting.")
        return

    # 获取嵌入权重
    embeddings = net.embedding.weight.data.cpu().numpy() # [N_prods, 2]
    prod_names = [k for k, v in sorted(dataset.prod_map.items(), key=lambda item: item[1])]

    plt.figure(figsize=(8, 6))
    
    # 绘制所有点
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', s=100, alpha=0.7)
    
    # 标注名称
    for i, name in enumerate(prod_names):
        color = 'red' if name == CONFIG["target_product"] else 'black'
        weight = 'bold' if name == CONFIG["target_product"] else 'normal'
        plt.text(embeddings[i, 0]+0.02, embeddings[i, 1]+0.02, name, 
                 fontsize=12, color=color, fontweight=weight)
        
        if name == CONFIG["target_product"]:
            plt.scatter(embeddings[i, 0], embeddings[i, 1], c='red', s=150, marker='*', label='Target (New)')

    plt.title(f"Learned Product Embeddings (2D)\nTarget: {CONFIG['target_product']}", fontsize=14)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Embedding plot saved to {save_path}")

def visualize_prediction(net, dataset, stats, exp_id, save_path=None):
    # (Same as before, ensure net is passed correctly)
    net.eval()
    exp_data = dataset.get_exp_events(exp_id)
    t_start, t_end = exp_data['times'][0].item(), exp_data['times'][-1].item()
    t_dense = torch.linspace(t_start, t_end, 200).to(CONFIG['device'])
    
    with torch.no_grad():
        pred_y = solve_with_events(net, exp_data, stats, t_eval_points=t_dense)
    t_pred, y_pred = t_dense.cpu().numpy(), pred_y.cpu().numpy()
    
    t_true = exp_data['times'].cpu().numpy()
    y_true = exp_data['targets'].cpu().numpy()
    mask_up = exp_data['mask'].cpu().numpy()
    
    cols = dataset.state_cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(t_pred, y_pred[:, i], label='HR Model', color='tab:blue')
        mi = mask_up[:, i].astype(bool)
        ax.scatter(t_true[mi], y_true[mi, i], label='Exp Data', color='tab:red', zorder=5)
        ax.set_title(col)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i==0: ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

# ==========================================
# 9. Main Task Manager
# ==========================================
def main_task_manager(task_type="train", data_path="data/real_world_data.csv", model_path="best_mtl_model.pth"):
    device = CONFIG["device"]

    if task_type == "train":
        print(f"\n>>> Multi-Task Training: Target {CONFIG['target_product']}")
        model, ds, stats = train_pipeline(file_path=data_path)
        
        # 1. 评估在未见过的 HP5 测试集上的表现
        print("\n--- Evaluation on Unseen Target Runs ---")
        test_avg, test_vec = run_evaluation(model, ds, stats, ds.test_ids)
        print(f"Test NRMSE (Avg): {test_avg:.2f}")
        for i, c in enumerate(ds.state_cols):
            print(f"  {c}: {test_vec[i]:.2f}")
            
        # 2. 可视化 HP5 测试集的预测
        for sid in ds.test_ids[:3]:
            visualize_prediction(model, ds, stats, sid, save_path=f"test_pred_{sid}.png")
            
        # 3. [New] 可视化 Embedding 空间
        plot_product_embeddings(model, ds)

    elif task_type == "evaluate":
        print(f"\n>>> Starting Task: EVALUATION using {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} not found!"); return
        
        # 1. 加载 Checkpoint 以获取 stats
        checkpoint = torch.load(model_path, map_location=device)
        stats = checkpoint['stats']
        
        # 2. 加载数据集（用于获取 num_products 和 test_ids）
        # 注意：这里我们加载全量数据，但只在 test_ids 上评估
        ds = BioreactorDataset(data_path) 
        
        # 3. 初始化网络 (必须包含 Embedding 参数)
        net = KineticsNet(input_dim=9, output_dim=6, 
                          num_products=ds.num_products, 
                          embedding_dim=CONFIG["embedding_dim"]).to(device)
        
        # 4. 加载权重
        net.load_state_dict(checkpoint['model_state_dict'])
        
        # 5. 执行评估
        test_avg, test_vec = run_evaluation(net, ds, stats, ds.test_ids)
        print(f"NRMSE on {data_path} (Test Set): {test_avg:.2f}")
        for i, c in enumerate(ds.state_cols):
            print(f"  {c}: {test_vec[i]:.2f}")

        # 6. 可视化部分测试样本
        # 确保 test_ids 不为空
        if len(ds.test_ids) > 0:
            for sid in ds.test_ids[:3]:
                visualize_prediction(net, ds, stats, sid, save_path=f"eval_{sid}.png")
                
        # 7. 可视化 Embedding
        plot_product_embeddings(net, ds, save_path="eval_embedding.png")

    elif task_type == "simulation":
        print(f"\n>>> Starting Task: SIMULATION using {data_path}")
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} not found!"); return

        # 1. 加载 Checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        train_stats = checkpoint['stats']
        
        # 2. 实例化数据集
        sim_ds = BioreactorDataset(data_path)
        
        # 3. 初始化网络 (关键：num_products 必须与训练时一致)
        # 注意：如果 simulation 的 csv 包含全新的 Product ID，
        # 直接加载旧权重会报错（Embedding 索引越界）。
        # 假设 simulation 的产品集合是训练集的子集，或者是已知的 ID。
        net = KineticsNet(input_dim=9, output_dim=6, 
                          num_products=sim_ds.num_products, 
                          embedding_dim=CONFIG["embedding_dim"]).to(device)
        
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        # 4. 执行模拟并绘图
        print(f"Simulating {len(sim_ds.all_exp_ids)} experiments...")
        for exp_id in sim_ds.all_exp_ids:
            # 内部 visualize_prediction 已支持 Mask，无真实点时自动只画线
            visualize_prediction(net, sim_ds, train_stats, exp_id, 
                                 save_path=f"sim_{exp_id}.png")
        
        print("Simulation complete.")
if __name__ == "__main__":
    # 假设你的CSV里现在有了 Product_ID 列
    if os.path.exists("data/hist.csv"):
        main_task_manager(task_type="train", data_path="data/hist.csv")
    else:
        print("Data file not found.")