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
    """Fix random seeds for reproducibility."""
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
    "hidden_dim": 32,
    "lr": 0.002,  # Start slightly higher, rely on scheduler
    "epochs": 500,  # Max epochs
    "patience": 20,  # Early stopping patience
    "batch_size": 2,  # Gradient accumulation steps (simulated batch size)
    "split_ratio": [0.1, 0.1, 0.8],  # Train / Val / Test
    "device": torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "print_freq": 1,
    "ode_method": "rk4",
    "loss_type": "relative_mse",
}

print(f"Using device: {CONFIG['device']}")


# ==========================================
# 2. Dataset Handler (Event-Driven + Mask Support)
# ==========================================
class BioreactorDataset:
    def __init__(self, csv_path, split_ratio=[0.8, 0.1, 0.1]):
        self.df = pd.read_csv(csv_path)
        self.state_cols = ["VCD", "Glc", "Gln", "Amm", "Lac", "product"]
        self.input_cols = ["Cmd_Temp", "Cmd_pH", "Cmd_Stirring"]
        self.vol_col = "Volume"
        self.feed_glc_mass_col = "Cmd_Feed_Glc_Mass"
        self.feed_gln_mass_col = "Cmd_Feed_Gln_Mass"
        self.feed_vol_col = "Cmd_Feed_Vol"

        self.all_exp_ids = self.df["Exp_ID"].unique()
        n_total = len(self.all_exp_ids)
        shuffled_ids = np.random.permutation(self.all_exp_ids)

        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        self.train_ids = shuffled_ids[:n_train]
        self.val_ids = shuffled_ids[n_train : n_train + n_val]
        self.test_ids = shuffled_ids[n_train + n_val :]

        if len(self.test_ids) == 0 and n_total > 1:
            self.test_ids = shuffled_ids[-1:]
            self.train_ids = shuffled_ids[:-1]
            self.val_ids = []
            print("Warning: Dataset too small.")

        print(
            f"Split -> Train: {len(self.train_ids)}, Val: {len(self.val_ids)}, Test: {len(self.test_ids)}"
        )

        # --- Global Statistics (Computed on Train set only, ignoring NaNs) ---
        train_df = self.df[self.df["Exp_ID"].isin(self.train_ids)]

        # Calculate mean/std ignoring NaNs
        self.state_mean = torch.tensor(
            train_df[self.state_cols].mean().values, dtype=torch.float32
        )
        self.state_std = torch.tensor(
            train_df[self.state_cols].std().values + 1e-6, dtype=torch.float32
        )

        self.input_mean = torch.tensor(
            train_df[self.input_cols].mean().values, dtype=torch.float32
        )
        self.input_std = torch.tensor(
            train_df[self.input_cols].std().values + 1e-6, dtype=torch.float32
        )

    def get_exp_events(self, exp_id):
        exp_df = self.df[self.df["Exp_ID"] == exp_id].sort_values("time[h]")
        t_np = exp_df["time[h]"].values.astype(np.float32)

        # Targets
        targets_np = exp_df[self.state_cols].values.astype(np.float32)
        mask_np = ~np.isnan(targets_np)
        targets = torch.tensor(np.nan_to_num(targets_np, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32)

        # Initial condition (Handle NaN in y0: fill with mean if missing, though rare for t=0)
        y0 = targets[0].clone()

        input_funcs = []
        for col in self.input_cols:
            f = interp1d(
                t_np, exp_df[col].values, kind="previous", fill_value="extrapolate"
            )
            input_funcs.append(f)

        events = []
        for _, row in exp_df.iterrows():
            t = float(row["time[h]"])
            glc_m = (
                float(row[self.feed_glc_mass_col])
                if not pd.isna(row[self.feed_glc_mass_col])
                else 0.0
            )
            gln_m = (
                float(row[self.feed_gln_mass_col])
                if not pd.isna(row[self.feed_gln_mass_col])
                else 0.0
            )
            f_vol = (
                float(row[self.feed_vol_col])
                if not pd.isna(row[self.feed_vol_col])
                else 0.0
            )

            if glc_m > 1e-6 or gln_m > 1e-6 or f_vol > 1e-6:
                events.append(
                    {"time": t, "glc_mass": glc_m, "gln_mass": gln_m, "vol_add": f_vol}
                )

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
            "exp_id": exp_id,
        }


# ==========================================
# 3. Hybrid Model Components (Unchanged)
# ==========================================
class KineticsNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 7),
            nn.Tanh(),
            nn.Linear(7, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last_layer = self.net[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=1e-3)
        nn.init.zeros_(last_layer.bias)

    def forward(self, x):
        return self.net(x)


class ReactionODEFunc(nn.Module):
    def __init__(self, kinetics_net, input_funcs, stats):
        super().__init__()
        self.net = kinetics_net
        self.input_funcs = input_funcs
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
        nn_input = torch.cat([norm_y, norm_controls], dim=-1)

        rates = self.net(nn_input)
        mu, q_glc, q_gln, q_amm, q_lac, q_prod = (
            rates[0],
            rates[1],
            rates[2],
            rates[3],
            rates[4],
            rates[5],
        )

        X_v = y[0]
        X_v_safe = torch.relu(X_v)

        d_vcd = mu * X_v_safe
        d_glc = q_glc * X_v_safe
        d_gln = q_gln * X_v_safe
        d_amm = q_amm * X_v_safe
        d_lac = q_lac * X_v_safe
        d_prod = q_prod * X_v_safe
        return torch.stack([d_vcd, d_glc, d_gln, d_amm, d_lac, d_prod])


# ==========================================
# 4. Solver (Unchanged)
# ==========================================
def solve_with_events(net, exp_data, stats, t_eval_points=None):
    device = CONFIG["device"]
    y_current = exp_data["y0"].to(device)
    current_vol = exp_data["initial_vol"]

    if t_eval_points is None:
        t_targets = exp_data["times"].cpu().numpy()
    else:
        t_targets = t_eval_points.cpu().numpy()

    events = exp_data["events"]
    feed_times = [e["time"] for e in events]
    all_stops = sorted(list(set(np.concatenate([t_targets, feed_times]))))

    pred_dict = {}
    # Store initial state if t=0 matches target
    if abs(all_stops[0] - t_targets[0]) < 1e-4:
        pred_dict[all_stops[0]] = y_current.clone()

    ode_func = ReactionODEFunc(net, exp_data["input_funcs"], stats).to(device)
    current_t = all_stops[0]

    for next_t in all_stops[1:]:
        if next_t <= current_t:
            continue
        t_span = torch.tensor([current_t, next_t], dtype=torch.float32).to(device)
        sol = odeint(ode_func, y_current, t_span, method=CONFIG["ode_method"], rtol=1e-5, atol=1e-7)
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
            y_next = torch.stack(
                [
                    total_vcd / current_vol,
                    total_glc / current_vol,
                    total_gln / current_vol,
                    total_amm / current_vol,
                    total_lac / current_vol,
                    total_prod / current_vol,
                ]
            )

        if np.any(np.isclose(t_targets, next_t, atol=1e-4)):
            pred_dict[next_t] = y_next.clone()
        current_t = next_t
        y_current = y_next

    final_preds = []
    for t in t_targets:
        # Robust lookup
        closest_t = min(pred_dict.keys(), key=lambda x: abs(x - t))
        final_preds.append(pred_dict[closest_t])
    return torch.stack(final_preds)


# ==========================================
# 5. Helper Class: Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=20, delta=0, save_path="best_hr_model.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


# ==========================================
# 6. Loss Functions with Mask Support [New]
# ==========================================
def masked_mse(pred, target, mask, reduction="mean"):
    """
    Calculate pure MSE, ignoring NaNs in target.
    """

    # If a variable is all NaN, mask sum is 0, avoid div by 0
    diff = (pred - target) * mask
    n_eff = torch.clamp(mask.sum(dim=0), min=1.0)  # [D]
    mse_vec = (diff**2).sum(dim=0) / n_eff

    if reduction == "none":
        return mse_vec
    elif reduction == "mean":
        return torch.mean(mse_vec)
    else:
        raise ValueError("reduction must be 'none' or 'mean'")


def masked_relative_mse(pred, target, mask, std_vec):
    """
    Calculate Relative MSE (Standardized), ignoring NaNs.
    Uses GLOBAL STD from training set for scaling (more stable than batch std with NaNs).
    """
    eps = 1e-6
    # Scale difference by global std
    diff = (pred - target) * mask  # [T, D]
    n_eff = torch.clamp(mask.sum(dim=0), min=1.0)  # [D]
    mse = (diff**2).sum(dim=0) / n_eff  # [D]
    rmse = torch.sqrt(mse)
    std = torch.clamp(std_vec, min=eps)
    return rmse / std


# ==========================================
# 7. Training Pipeline (Modified)
# ==========================================
def run_evaluation(net, dataset, stats, target_ids):
    if len(target_ids) == 0:
        return 0.0, np.zeros(6)
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
                nrmse_vec = masked_relative_mse(pred_y, targets, mask, stats["state_std"])

                total_nrmse_vec += nrmse_vec
                count += 1
            except Exception as e:
                print(f"Error evaluating {exp_id}: {e}")

    avg_nrmse_vec = total_nrmse_vec / max(count, 1)
    avg_scalar = avg_nrmse_vec.mean().item()
    return avg_scalar, avg_nrmse_vec.cpu().numpy()


def train_pipeline(file_path="data/test.csv"):
    dataset = BioreactorDataset(file_path, split_ratio=CONFIG["split_ratio"])
    stats = {
        "state_mean": dataset.state_mean.to(CONFIG["device"]),
        "state_std": dataset.state_std.to(CONFIG["device"]),
        "input_mean": dataset.input_mean.to(CONFIG["device"]),
        "input_std": dataset.input_std.to(CONFIG["device"]),
    }

    net = KineticsNet(input_dim=9, output_dim=6).to(CONFIG["device"])

    optimizer = optim.Adam(net.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    early_stopping = EarlyStopping(
        patience=CONFIG["patience"], save_path="best_hr_model.pth"
    )

    print(f"\n--- Starting Training (Max Epochs: {CONFIG['epochs']}) ---")

    history = {"train_loss": [], "val_nrmse_avg": [], "val_nrmse_vars": []}
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
            pred_y = solve_with_events(net, exp_data, stats)

            # Loss Calculation with Mask
            if CONFIG["loss_type"] == "mse":
                loss = masked_mse(pred_y, targets, mask)
            else:
                nrmse_vec = masked_relative_mse(
                    pred_y, targets, mask, stats["state_std"]
                )
                loss = torch.mean(nrmse_vec)


            # Gradient Accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_train_ids):
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

            batch_loss += loss.item() * accumulation_steps

        avg_loss = batch_loss / len(epoch_train_ids)
        history["train_loss"].append(avg_loss)

        # Validation
        if (epoch + 1) % CONFIG["print_freq"] == 0:
            val_avg, val_vec = run_evaluation(net, dataset, stats, dataset.val_ids)
            history["val_nrmse_avg"].append(val_avg)
            history["val_nrmse_vars"].append(val_vec)

            print(
                f"Epoch {epoch+1:04d} | Loss: {avg_loss:.5f} | Val Avg: {val_avg:.2f} | Prod: {val_vec[5]:.2f}"
            )

            scheduler.step(val_avg)
            early_stopping(val_avg, net)
            if early_stopping.early_stop:
                print("--- Early stopping triggered ---")
                break

    print(f"\nLoading best model from {early_stopping.save_path}...")
    net.load_state_dict(torch.load(early_stopping.save_path))
    return net, dataset, stats, history


# ==========================================
# 8. Visualization (Mask Supported)
# ==========================================
def plot_training_metrics(history, state_cols):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    val_epochs = np.arange(1, len(history["val_nrmse_avg"]) + 1)
    if len(val_epochs) != len(epochs):
        val_epochs = np.linspace(1, len(epochs), len(history["val_nrmse_avg"]))

    val_vars_np = np.array(history["val_nrmse_vars"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(
        epochs, history["train_loss"], label="Train Loss", color="black", alpha=0.6
    )
    ax1_r = ax1.twinx()
    ax1_r.plot(
        val_epochs,
        history["val_nrmse_avg"],
        label="Val Avg NRMSE",
        color="red",
        linewidth=2,
    )
    ax1.set_title("Training Loss & Avg NRMSE")
    ax1.legend(loc="upper left")
    ax1_r.legend(loc="upper right")

    colors = plt.cm.tab10(np.linspace(0, 1, len(state_cols)))
    if len(val_epochs) > 0:
        for i, col_name in enumerate(state_cols):
            ax2.plot(
                val_epochs,
                val_vars_np[:, i],
                label=col_name,
                color=colors[i],
                linewidth=2,
            )
    ax2.set_title("NRMSE per Variable")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("training_metrics_detailed.png", dpi=150)


def visualize_prediction(net, dataset, stats, exp_id, save_path=None):
    net.eval()
    exp_data = dataset.get_exp_events(exp_id)
    t_start, t_end = exp_data["times"][0].item(), exp_data["times"][-1].item()
    t_dense = torch.linspace(t_start, t_end, 200).to(CONFIG["device"])

    with torch.no_grad():
        pred_y = solve_with_events(net, exp_data, stats, t_eval_points=t_dense)
    t_pred, y_pred = t_dense.cpu().numpy(), pred_y.cpu().numpy()

    # Ground truth (handle NaNs for plotting)
    t_true = exp_data["times"].cpu().numpy()
    y_true = exp_data["targets"].cpu().numpy()

    cols = dataset.state_cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(t_pred, y_pred[:, i], label="HR Model", color="tab:blue")

        # Only plot valid points (not NaN)
        valid_mask = ~np.isnan(y_true[:, i])
        if np.any(valid_mask):
            ax.scatter(
                t_true[valid_mask],
                y_true[valid_mask, i],
                label="Exp Data",
                color="tab:red",
                zorder=5,
            )

        ax.set_title(col)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    if os.path.exists("data/real_world_data.csv"):
        # Train
        model, ds, stats, history = train_pipeline(file_path="data/real_world_data.csv")

        # Plot Metrics
        plot_training_metrics(history, ds.state_cols)

        # Eval
        print("\n--- Final Evaluation on Test Set ---")
        test_avg, test_vec = run_evaluation(model, ds, stats, ds.test_ids)
        print(f"Final Test NRMSE (Avg): {test_avg:.2f}")
        for i, c in enumerate(ds.state_cols):
            print(f"  {c}: {test_vec[i]:.2f}")

        # Visualize
        if len(ds.test_ids) > 0:
            for i in range(min(10, len(ds.test_ids))):
                sid = ds.test_ids[i]
                print(f"Visualizing {sid}...")
                visualize_prediction(model, ds, stats, sid, save_path=f"pred_{sid}.png")
    else:
        print("Error: 'novel.csv' not found.")
