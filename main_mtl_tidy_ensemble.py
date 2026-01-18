"""
Multi-Task Ensemble Neural ODE Model for Bioreactor Simulation

This module implements a hybrid physics-informed neural network with:
1. Product embeddings for multi-task learning (Transfer Learning).
2. Ensemble methods for uncertainty quantification (Risk Assessment).
3. Dynamic feature engineering to support future scale-up parameters.

Key Features:
    - Ensemble Learning (N=5) for Confidence Intervals
    - Auto-detection of Engineering Parameters (e.g., P/V, Tip Speed)
    - Few-shot learning for new product adaptation
"""

import os
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torchdiffeq import odeint


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class Config:
    """Configuration for model training and inference."""

    # Model architecture
    hidden_dim: int = 64
    embedding_dim: int = 2

    # --- [NEW] Uncertainty Quantification ---
    ensemble_size: int = 5  # Number of models to train for voting
    uncertainty_k: float = 2.0  # Sigma multiplier for visualization (2.0 ~= 95% CI)

    # Multi-task settings
    target_product: str = "NONE"
    n_target_shots: int = 0

    # Single-task settings
    single_prod_train_ratio: float = 0.1
    single_prod_val_ratio: float = 0.1

    # Training hyperparameters
    learning_rate: float = 0.002
    epochs: int = 150  # Slightly reduced as we train N models
    patience: int = 15
    batch_size: int = 4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.5

    # ODE solver settings
    ode_method: str = "euler"
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7

    # Paths
    model_save_path: str = "best_ensemble_model.pth"

    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CONFIG = Config()
# Note: For ensemble, we might want *different* seeds for initialization,
# but same seeds for data splitting. Handled in logic below.
set_random_seed(42)
print(f"Using device: {CONFIG.device}")


# ==============================================================================
# Dataset Handler (Upgraded for Engineering Params)
# ==============================================================================


@dataclass
class ExperimentData:
    times: torch.Tensor
    y0: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    input_funcs: List[Callable]
    events: List[Dict[str, float]]
    initial_vol: float
    exp_id: str
    product_idx: torch.Tensor


@dataclass
class NormalizationStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def to_device(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "state_mean": self.state_mean.to(device),
            "state_std": self.state_std.to(device),
            "input_mean": self.input_mean.to(device),
            "input_std": self.input_std.to(device),
        }


class BioreactorDataset:
    """
    Dataset handler with dynamic support for Engineering Parameters.
    """

    # Standard columns
    STATE_COLS = ["VCD", "Glc", "Gln", "Amm", "Lac", "product"]
    BASE_INPUT_COLS = ["Cmd_Temp", "Cmd_pH", "Cmd_Stirring"]

    # --- [NEW] Scale-Up / Engineering Parameter Placeholders ---
    # If these columns exist in the CSV, they will be automatically used.
    # If not, they are ignored.
    OPTIONAL_ENG_COLS = [
        "Power_Input",  # P/V (W/m3)
        "Tip_Speed",  # m/s
        "kLa",  # 1/h
        "VVM",  # Aeration rate
        "Headspace_Pres",  # Pressure
    ]

    VOL_COL = "Volume"
    FEED_GLC_COL = "Cmd_Feed_Glc_Mass"
    FEED_GLN_COL = "Cmd_Feed_Gln_Mass"
    FEED_VOL_COL = "Cmd_Feed_Vol"
    PROD_ID_COL = "Product_ID"
    TIME_COL = "time[h]"
    EXP_ID_COL = "Exp_ID"

    def __init__(
        self,
        csv_path: str,
        config: Config = CONFIG,
        existing_prod_map: Optional[Dict] = None,
    ):
        self.config = config
        self.df = pd.read_csv(csv_path)

        if self.PROD_ID_COL not in self.df.columns:
            print("Info: No Product_ID found. Auto-filling as 'Default_Product'.")
            self.df[self.PROD_ID_COL] = "Default_Product"

        # --- [NEW] Dynamic Input Column Detection ---
        self.active_input_cols = list(self.BASE_INPUT_COLS)
        found_eng_cols = []
        for col in self.OPTIONAL_ENG_COLS:
            if col in self.df.columns:
                self.active_input_cols.append(col)
                found_eng_cols.append(col)

        if found_eng_cols:
            print(f"--> [Scale-Up] Detected Engineering Parameters: {found_eng_cols}")
        else:
            print("--> [Info] No Engineering Parameters found. Using standard inputs.")

        self.input_dim = len(self.active_input_cols)
        # --------------------------------------------

        self._create_unique_ids()
        self._build_product_mapping(existing_prod_map)
        self._split_data()
        self._compute_statistics()

    # ... (Other methods: _create_unique_ids, _build_product_mapping, _split_data unchanged) ...

    def _create_unique_ids(self) -> None:
        self.df["Unique_ID"] = (
            self.df[self.PROD_ID_COL].astype(str)
            + "_"
            + self.df[self.EXP_ID_COL].astype(str)
        )

    def _build_product_mapping(self, existing_map: Optional[Dict]) -> None:
        current_products = sorted(self.df[self.PROD_ID_COL].unique())

        if existing_map is not None:
            self.prod_map = existing_map.copy()
            next_idx = max(self.prod_map.values()) + 1
            for name in current_products:
                if name not in self.prod_map:
                    self.prod_map[name] = next_idx
                    next_idx += 1
        else:
            self.prod_map = {name: idx for idx, name in enumerate(current_products)}

        self.num_products = len(self.prod_map)
        print(f"Product Mapping: {self.prod_map}")

    def _split_data(self) -> None:
        all_exp_ids = self.df["Unique_ID"].unique()
        if self.num_products > 1:
            self._split_multi_product_few_shot(all_exp_ids)
        else:
            self._split_single_product_random(all_exp_ids)
        print(
            f"Dataset Split: Train={len(self.train_ids)}, Val={len(self.val_ids)}, Test={len(self.test_ids)}"
        )

    def _split_multi_product_few_shot(self, all_exp_ids):
        # ... (Same as before) ...
        exp_to_product = (
            self.df.groupby("Unique_ID")[self.PROD_ID_COL].first().to_dict()
        )
        historical_exps = []
        target_exps = []
        target_name = self.config.target_product

        for uid in all_exp_ids:
            if exp_to_product.get(uid) == target_name:
                target_exps.append(uid)
            else:
                historical_exps.append(uid)

        random.shuffle(target_exps)
        train_target = target_exps[: self.config.n_target_shots]
        test_target = target_exps[self.config.n_target_shots :]

        train_pool = np.concatenate([historical_exps, train_target])
        np.random.shuffle(train_pool)

        n_val = max(1, int(len(train_pool) * 0.1))
        self.val_ids = train_pool[:n_val]
        self.train_ids = train_pool[n_val:]
        self.test_ids = np.array(test_target)
        self.all_exp_ids = all_exp_ids

    def _split_single_product_random(self, all_exp_ids):
        # ... (Same as before) ...
        np.random.shuffle(all_exp_ids)
        n_total = len(all_exp_ids)
        n_train = int(n_total * self.config.single_prod_train_ratio)
        n_val = int(n_total * self.config.single_prod_val_ratio)

        self.train_ids = all_exp_ids[:n_train]
        self.val_ids = all_exp_ids[n_train : n_train + n_val]
        self.test_ids = all_exp_ids[n_train + n_val :]
        self.all_exp_ids = all_exp_ids

    def _compute_statistics(self) -> None:
        """Compute normalization stats including potentially new Engineering cols."""
        train_df = self.df[self.df["Unique_ID"].isin(self.train_ids)]

        state_mean = train_df[self.STATE_COLS].mean().values
        state_std = train_df[self.STATE_COLS].std().values + 1e-6

        # Use active_input_cols (Dynamic)
        input_mean = train_df[self.active_input_cols].mean().values
        input_std = train_df[self.active_input_cols].std().values + 1e-6

        self.state_mean = torch.tensor(
            np.nan_to_num(state_mean, nan=0.0), dtype=torch.float32
        )
        self.state_std = torch.tensor(
            np.nan_to_num(state_std, nan=1.0), dtype=torch.float32
        )
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)
        self.input_std = torch.tensor(input_std, dtype=torch.float32)

    def get_normalization_stats(self) -> NormalizationStats:
        return NormalizationStats(
            self.state_mean, self.state_std, self.input_mean, self.input_std
        )

    def get_experiment_data(self, unique_id: str) -> ExperimentData:
        exp_df = self.df[self.df["Unique_ID"] == unique_id].sort_values(self.TIME_COL)
        times = exp_df[self.TIME_COL].values.astype(np.float32)
        product_name = exp_df[self.PROD_ID_COL].iloc[0]
        product_idx = self.prod_map[product_name]

        targets = torch.tensor(
            np.nan_to_num(exp_df[self.STATE_COLS].values, nan=0.0), dtype=torch.float32
        )
        mask = torch.tensor(
            (~np.isnan(exp_df[self.STATE_COLS].values)).astype(np.float32),
            dtype=torch.float32,
        )
        y0 = self._compute_initial_state(targets)
        input_funcs = self._build_input_functions(exp_df, times)
        events = self._extract_feed_events(exp_df)
        initial_vol = float(exp_df.iloc[0][self.VOL_COL])

        return ExperimentData(
            torch.tensor(times),
            y0,
            targets,
            mask,
            input_funcs,
            events,
            initial_vol,
            unique_id,
            torch.tensor(product_idx, dtype=torch.long),
        )

    def _compute_initial_state(self, targets: torch.Tensor) -> torch.Tensor:
        y0 = targets[0].clone()
        for i in range(len(y0)):
            if torch.isnan(y0[i]) or y0[i] == 0:
                valid = targets[:, i][targets[:, i] > 0]
                if len(valid) > 0:
                    y0[i] = valid[0]
        return y0

    def _build_input_functions(
        self, exp_df: pd.DataFrame, times: np.ndarray
    ) -> List[Callable]:
        input_funcs = []
        # Dynamic inputs
        for col in self.active_input_cols:
            values = exp_df[col].interpolate().bfill().ffill().values
            values = np.nan_to_num(values, nan=0.0)
            func = interp1d(times, values, kind="previous", fill_value="extrapolate")
            input_funcs.append(func)
        return input_funcs

    def _extract_feed_events(self, exp_df: pd.DataFrame) -> List[Dict[str, float]]:
        events = []
        for _, row in exp_df.iterrows():
            glc = float(row.get(self.FEED_GLC_COL, 0))
            gln = float(row.get(self.FEED_GLN_COL, 0))
            vol = float(row.get(self.FEED_VOL_COL, 0))
            if glc > 1e-6 or gln > 1e-6 or vol > 1e-6:
                events.append(
                    {
                        "time": float(row[self.TIME_COL]),
                        "glc_mass": glc,
                        "gln_mass": gln,
                        "vol_add": vol,
                    }
                )
        return sorted(events, key=lambda x: x["time"])


# ==============================================================================
# Core Models (KineticsNet & ODE)
# ==============================================================================


class KineticsNet(nn.Module):
    """Neural network with expandable embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_products: int,
        embedding_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_products, embedding_dim)
        total_input_dim = input_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        nn.init.normal_(self.net[-1].weight, 0, 1e-4)

    def forward(self, x: torch.Tensor, product_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(product_idx)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).expand(x.size(0), -1)
        combined = torch.cat([x, emb], dim=-1)
        return self.net(combined)

    def expand_embeddings(self, new_num_products: int):
        old_emb = self.embedding
        old_num, dim = old_emb.weight.shape
        if new_num_products <= old_num:
            return
        print(f"Expanding embedding: {old_num} -> {new_num_products}")
        new_emb = nn.Embedding(new_num_products, dim).to(old_emb.weight.device)
        with torch.no_grad():
            new_emb.weight[:old_num] = old_emb.weight
            mean_emb = old_emb.weight.mean(dim=0)
            for i in range(old_num, new_num_products):
                new_emb.weight[i] = mean_emb + torch.randn_like(mean_emb) * 0.05
        self.embedding = new_emb


class ReactionODEFunc(nn.Module):
    def __init__(self, kinetics_net, input_funcs, stats, product_idx):
        super().__init__()
        self.net = kinetics_net
        self.input_funcs = input_funcs
        self.product_idx = product_idx
        self.register_buffer("state_mean", stats["state_mean"])
        self.register_buffer("state_std", stats["state_std"])
        self.register_buffer("input_mean", stats["input_mean"])
        self.register_buffer("input_std", stats["input_std"])
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_val = t.item()
        
        # --- [修复开始] ---
        # 1. 先转为 numpy array (处理 interp1d 返回的 0-d array 问题)
        controls_np = np.array([f(t_val) for f in self.input_funcs], dtype=np.float32)
        # 2. 再转为 tensor 并移至对应设备
        controls = torch.from_numpy(controls_np).to(y.device)
        # --- [修复结束] ---
        
        # Normalize
        norm_y = (y - self.state_mean) / self.state_std
        norm_controls = (controls - self.input_mean) / self.input_std
        
        # Exclude titer from inputs (index 5)
        nn_input = torch.cat([norm_y[:5], norm_controls], dim=-1)
        rates = self.net(nn_input.unsqueeze(0), self.product_idx).squeeze(0)
        
        mu, q_glc, q_gln, q_amm, q_lac, q_prod = rates
        x_v = torch.nn.functional.softplus(y[0])
        
        d_vcd = mu * x_v
        d_glc = -q_glc * x_v
        d_gln = -q_gln * x_v
        d_amm = q_amm * x_v
        d_lac = q_lac * x_v
        d_prod = q_prod * x_v
        
        return torch.stack([d_vcd, d_glc, d_gln, d_amm, d_lac, d_prod])


# ==============================================================================
# Solver
# ==============================================================================


def solve_with_events(net, exp_data, stats, t_eval_points=None, config=CONFIG):
    device = config.device
    y_curr = exp_data.y0.to(device)
    curr_vol = exp_data.initial_vol

    if t_eval_points is None:
        t_targets = exp_data.times.cpu().numpy()
    else:
        t_targets = t_eval_points.cpu().numpy()

    feed_times = [e["time"] for e in exp_data.events]
    all_stops = sorted(set(np.concatenate([t_targets, feed_times])))

    pred_dict = {}
    if abs(all_stops[0] - t_targets[0]) < 1e-4:
        pred_dict[all_stops[0]] = y_curr.clone()

    ode_func = ReactionODEFunc(
        net, exp_data.input_funcs, stats, exp_data.product_idx.to(device)
    )
    curr_t = all_stops[0]

    for next_t in all_stops[1:]:
        if next_t <= curr_t:
            continue
        t_span = torch.tensor([curr_t, next_t], dtype=torch.float32, device=device)
        sol = odeint(
            ode_func,
            y_curr,
            t_span,
            method=config.ode_method,
            rtol=config.ode_rtol,
            atol=config.ode_atol,
        )
        y_next = sol[-1]

        # Apply Feed
        events_now = [e for e in exp_data.events if abs(e["time"] - next_t) < 1e-4]
        for ev in events_now:
            masses = y_next * curr_vol
            masses[1] += ev["glc_mass"]
            masses[2] += ev["gln_mass"]
            curr_vol += ev["vol_add"]
            y_next = masses / curr_vol

        if np.any(np.isclose(t_targets, next_t, atol=1e-4)):
            pred_dict[next_t] = y_next.clone()

        curr_t = next_t
        y_curr = y_next

    final_preds = []
    for t in t_targets:
        closest_t = min(pred_dict.keys(), key=lambda x: abs(x - t))
        final_preds.append(pred_dict[closest_t])
    return torch.stack(final_preds)


# ==============================================================================
# Ensemble Training & Inference
# ==============================================================================


def masked_relative_mse(pred, target, mask, std, eps=1e-6):
    diff = (pred - target) * mask
    mse = (diff**2).sum(dim=0) / torch.clamp(mask.sum(dim=0), min=1.0)
    return torch.sqrt(mse) / torch.clamp(std, min=eps)


def train_ensemble(file_path: str, config: Config = CONFIG, existing_prod_map=None):
    """
    Train an ensemble of N models for uncertainty quantification.
    """
    print(f"\n>>> Starting Ensemble Training (N={config.ensemble_size})...")
    dataset = BioreactorDataset(file_path, config, existing_prod_map)
    stats = dataset.get_normalization_stats().to_device(config.device)

    ensemble_models = []

    for model_idx in range(config.ensemble_size):
        print(f"\n--- Training Model {model_idx + 1}/{config.ensemble_size} ---")

        # Initialize model with different random seed implicitly (via Linear init)
        net = KineticsNet(
            input_dim=5+dataset.input_dim,
            output_dim=6,
            num_products=dataset.num_products,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)

        optimizer = optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        best_loss = float("inf")
        patience_counter = 0

        # Training Loop
        for epoch in range(config.epochs):
            net.train()
            batch_loss = 0

            # Shuffle
            ids = np.random.permutation(dataset.train_ids)

            for exp_id in ids:
                exp_data = dataset.get_experiment_data(exp_id)
                targets = exp_data.targets.to(config.device)
                mask = exp_data.mask.to(config.device)

                try:
                    pred = solve_with_events(net, exp_data, stats, config=config)
                    loss = torch.mean(
                        masked_relative_mse(pred, targets, mask, stats["state_std"])
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), config.grad_clip_norm
                    )
                    optimizer.step()
                    batch_loss += loss.item()
                except:
                    continue

            # Validation
            if (epoch + 1) % 10 == 0:
                # Quick Val
                val_errs = []
                net.eval()
                for vid in dataset.val_ids:
                    vdata = dataset.get_experiment_data(vid)
                    with torch.no_grad():
                        vpred = solve_with_events(net, vdata, stats, config=config)
                        vloss = torch.mean(
                            masked_relative_mse(
                                vpred,
                                vdata.targets.to(config.device),
                                vdata.mask.to(config.device),
                                stats["state_std"],
                            )
                        )
                        val_errs.append(vloss.item())
                val_score = np.mean(val_errs)
                print(f"Ep {epoch+1} | Val Loss: {val_score:.4f}")

                if val_score < best_loss:
                    best_loss = val_score
                    patience_counter = 0
                    # Save best weights in memory temporarily
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print("Early stopping.")
                        break

        # Load best state for this member
        net.load_state_dict(best_state)
        ensemble_models.append(net)

    # Save Ensemble
    ensemble_states = [m.state_dict() for m in ensemble_models]
    save_dict = {
        "ensemble_states": ensemble_states,
        "stats": dataset.get_normalization_stats().to_device(
            torch.device("cpu")
        ),  # Save as CPU for portability
        "product_map": dataset.prod_map,
        "input_dim": dataset.input_dim,
    }
    torch.save(save_dict, config.model_save_path)
    print(f"Ensemble saved to {config.model_save_path}")

    return ensemble_models, dataset, stats


def predict_ensemble(
    ensemble_models: List[nn.Module],
    exp_data: ExperimentData,
    stats: Dict,
    t_eval: torch.Tensor,
    config: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on all models and calculate Mean and Std.
    Returns: (mean_pred, std_pred)
    """
    preds = []
    for net in ensemble_models:
        net.eval()
        with torch.no_grad():
            p = solve_with_events(
                net, exp_data, stats, t_eval_points=t_eval, config=config
            )
            preds.append(p)

    # Stack: [Ensemble, Time, Vars]
    preds_stack = torch.stack(preds)

    mean_pred = torch.mean(preds_stack, dim=0)
    std_pred = torch.std(preds_stack, dim=0)

    return mean_pred, std_pred


# ==============================================================================
# Visualization with Confidence Intervals
# ==============================================================================


def visualize_ensemble_prediction(
    ensemble_models: List[nn.Module],
    dataset: BioreactorDataset,
    stats: Dict,
    exp_id: str,
    save_path: str,
    config: Config,
):
    exp_data = dataset.get_experiment_data(exp_id)
    t_dense = torch.linspace(exp_data.times[0], exp_data.times[-1], 100).to(
        config.device
    )

    # Get Uncertainty
    mean_pred, std_pred = predict_ensemble(
        ensemble_models, exp_data, stats, t_dense, config
    )

    mean_np = mean_pred.cpu().numpy()
    std_np = std_pred.cpu().numpy()
    t_np = t_dense.cpu().numpy()

    obs_t = exp_data.times.cpu().numpy()
    obs_y = exp_data.targets.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(dataset.STATE_COLS):
        ax = axes[i]

        # Plot Mean
        ax.plot(
            t_np, mean_np[:, i], color="tab:blue", label="Mean Prediction", linewidth=2
        )

        # Plot Confidence Interval
        lower = mean_np[:, i] - config.uncertainty_k * std_np[:, i]
        upper = mean_np[:, i] + config.uncertainty_k * std_np[:, i]
        ax.fill_between(
            t_np,
            lower,
            upper,
            color="tab:blue",
            alpha=0.2,
            label=f"CI ($\pm${config.uncertainty_k}$\sigma$)",
        )

        # Plot Obs
        mask = exp_data.mask[:, i].bool().cpu().numpy()
        ax.scatter(
            obs_t[mask], obs_y[mask, i], color="tab:red", label="Observed", zorder=5
        )

        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.suptitle(f"Ensemble Prediction (N={config.ensemble_size}): {exp_id}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# ==============================================================================
# Fine-Tuning Task (Ensemble Aware)
# ==============================================================================


def fine_tune_ensemble_task(base_model_path: str, novel_data_path: str, config: Config):
    print(f"\n>>> Fine-Tuning Ensemble...")

    # 1. Load Base
    checkpoint = torch.load(base_model_path, map_location=config.device)
    base_states = checkpoint["ensemble_states"]
    base_stats = {k: v.to(config.device) for k, v in checkpoint["stats"].items()}
    base_map = checkpoint["product_map"]
    saved_control_dim = checkpoint.get("input_dim", 8)  # Fallback

    # 2. Dataset
    dataset = BioreactorDataset(novel_data_path, config, existing_prod_map=base_map)

    ft_models = []

    # 3. FT Loop for each model
    for idx, state_dict in enumerate(base_states):
        print(f"Fine-tuning member {idx+1}...")

        net = KineticsNet(
            input_dim=5+saved_control_dim,
            output_dim=6,
            num_products=len(base_map),  # Init with old size
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        ).to(config.device)

        net.load_state_dict(state_dict)
        net.expand_embeddings(dataset.num_products)  # Expand

        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate * 0.5)

        # Simple FT Loop
        for epoch in range(50):  # Fewer epochs for FT
            net.train()
            for exp_id in dataset.train_ids:
                exp_data = dataset.get_experiment_data(exp_id)
                try:
                    pred = solve_with_events(net, exp_data, base_stats, config=config)
                    loss = torch.mean(
                        masked_relative_mse(
                            pred,
                            exp_data.targets.to(config.device),
                            exp_data.mask.to(config.device),
                            base_stats["state_std"],
                        )
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except:
                    continue

        ft_models.append(net)

    # Visualize Test
    if len(dataset.test_ids) > 0:
        visualize_ensemble_prediction(
            ft_models,
            dataset,
            base_stats,
            dataset.test_ids[0],
            "ft_ensemble_vis.png",
            config,
        )


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    HIST_FILE = "data/hist.csv"
    NOVEL_FILE = "data/novel.csv"

    # Step 1: Train Base Ensemble if needed
    if os.path.exists(HIST_FILE):
        CONFIG.ensemble_size = 5
        models, _, _ = train_ensemble(HIST_FILE, CONFIG)

    # Step 2: Fine-tune if new data exists
    if os.path.exists(CONFIG.model_save_path) and os.path.exists(NOVEL_FILE):
        ft_config = copy.deepcopy(CONFIG)
        ft_config.n_target_shots = 2
        fine_tune_ensemble_task(CONFIG.model_save_path, NOVEL_FILE, ft_config)
