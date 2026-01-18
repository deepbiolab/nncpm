"""
Multi-Task Neural ODE Model for Bioreactor Simulation

This module implements a hybrid physics-informed neural network for modeling
bioreactor dynamics across multiple products using product embeddings.
Supports few-shot learning for new product adaptation.

Key Features:
    - Product embeddings for multi-task learning
    - Event-driven ODE solver with feed handling
    - Few-shot learning for new products
    - Comprehensive visualization tools
"""

import os
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

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
    embedding_dim: int = 2  # 2D for visualization
    
    # Multi-task settings
    target_product: str = "HP5"
    n_target_shots: int = 2  # Few-shot learning samples

    # Single-task settings
    single_prod_train_ratio: float = 0.1
    single_prod_val_ratio: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 0.002
    epochs: int = 200
    patience: int = 20
    batch_size: int = 4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.5
    
    # ODE solver settings
    ode_method: str = "rk4"
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-7
    
    # Logging
    print_freq: int = 1
    random_seed: int = 42
    
    # Loss configuration
    loss_type: str = "relative_mse"
    
    # Paths
    model_save_path: str = "best_mtl_model.pth"
    
    # Device (set dynamically)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    
    def __post_init__(self):
        """Set device after initialization."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Global configuration instance
CONFIG = Config()
set_random_seed(CONFIG.random_seed)
print(f"Using device: {CONFIG.device}")


# ==============================================================================
# Dataset Handler
# ==============================================================================

@dataclass
class ExperimentData:
    """Container for single experiment data."""
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
    """Container for normalization statistics."""
    state_mean: torch.Tensor
    state_std: torch.Tensor
    input_mean: torch.Tensor
    input_std: torch.Tensor
    
    def to_device(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Move all statistics to specified device."""
        return {
            "state_mean": self.state_mean.to(device),
            "state_std": self.state_std.to(device),
            "input_mean": self.input_mean.to(device),
            "input_std": self.input_std.to(device),
        }


class BioreactorDataset:
    """
    Dataset handler for bioreactor experiments with multi-product support.
    
    Handles data loading, preprocessing, and train/val/test splitting
    with few-shot learning support for target products.
    """
    
    # Column definitions
    STATE_COLS = ["VCD", "Glc", "Gln", "Amm", "Lac", "product"]
    INPUT_COLS = ["Cmd_Temp", "Cmd_pH", "Cmd_Stirring"]
    VOL_COL = "Volume"
    FEED_GLC_COL = "Cmd_Feed_Glc_Mass"
    FEED_GLN_COL = "Cmd_Feed_Gln_Mass"
    FEED_VOL_COL = "Cmd_Feed_Vol"
    PROD_ID_COL = "Product_ID"
    TIME_COL = "time[h]"
    EXP_ID_COL = "Exp_ID"
    
    def __init__(self, csv_path: str, config: Config = CONFIG, existing_prod_map: Optional[Dict] = None):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path: Path to the CSV data file.
            config: Configuration object.
        """
        self.config = config
        self.df = pd.read_csv(csv_path)

        if self.PROD_ID_COL not in self.df.columns:
            print("Info: No Product_ID found. Auto-filling as 'Default_Product'.")
            self.df[self.PROD_ID_COL] = "Default_Product"
        
        # Create unique experiment identifiers: "Product_ExpID"
        self._create_unique_ids()
        
        # Build product encoding
        self._build_product_mapping(existing_prod_map)
        
        # Split data into train/val/test
        self._split_data()
        
        # Compute normalization statistics from training data
        self._compute_statistics()
    
    def _create_unique_ids(self) -> None:
        """Create unique experiment identifiers combining product and exp ID."""
        self.df["Unique_ID"] = (
            self.df[self.PROD_ID_COL].astype(str) + "_" + 
            self.df[self.EXP_ID_COL].astype(str)
        )
    
    def _build_product_mapping(self, existing_map: Optional[Dict]) -> None:
        """Build mapping, optionally extending an existing one."""
        current_products = sorted(self.df[self.PROD_ID_COL].unique())
        
        if existing_map is not None:
            # 继承旧映射
            self.prod_map = existing_map.copy()
            next_idx = max(self.prod_map.values()) + 1
            
            # 检查是否有新产品
            self.new_products = []
            for name in current_products:
                if name not in self.prod_map:
                    print(f"New product detected: {name} -> assigned index {next_idx}")
                    self.prod_map[name] = next_idx
                    self.new_products.append(name)
                    next_idx += 1
        else:
            # 从头构建
            self.prod_map = {name: idx for idx, name in enumerate(current_products)}
            self.new_products = []
            
        self.num_products = len(self.prod_map)
        print(f"Product Mapping: {self.prod_map}")
    
    def _split_data(self) -> None:
        """
        Split data into train/val/test sets.
        
        Uses few-shot strategy: most target product data goes to test,
        only n_target_shots samples used for training.
        """
        all_exp_ids = self.df["Unique_ID"].unique()

        if self.num_products > 1:
            self._split_multi_product_few_shot(all_exp_ids)
        else:
            self._split_single_product_random(all_exp_ids)
        
        print(f"Dataset Split Summary (Products: {self.num_products}):")
        print(f"  Train: {len(self.train_ids)} runs")
        print(f"  Val  : {len(self.val_ids)} runs")
        print(f"  Test : {len(self.test_ids)} runs")

    def _split_multi_product_few_shot(self, all_exp_ids):
        """Original logic for multi-task few-shot learning."""
        # Map unique_id -> product_name
        exp_to_product = self.df.groupby("Unique_ID")[self.PROD_ID_COL].first().to_dict()
        
        historical_exps = []
        target_exps = []
        target_name = self.config.target_product
        
        for uid in all_exp_ids:
            if exp_to_product.get(uid) == target_name:
                target_exps.append(uid)
            else:
                historical_exps.append(uid)
        
        # Shuffle and split target experiments
        random.shuffle(target_exps)
        train_target = target_exps[:self.config.n_target_shots]
        test_target = target_exps[self.config.n_target_shots:]
        
        # Combine historical with few-shot target samples for training
        train_pool = np.concatenate([historical_exps, train_target])
        np.random.shuffle(train_pool)
        
        # 针对 Fine-tuning 场景的特殊处理：
        # 如果没有历史数据辅助，且总训练数据极少 (<= 3)，则不强制划分验证集
        if len(historical_exps) == 0 and len(train_pool) <= 3:
            print("Warning: Extremely low data regime for FT. Skipping Validation split to maximize Training.")
            self.train_ids = train_pool  # 全部用于训练
            self.val_ids = []            # 验证集为空
            
            # 为了防止 evaluate_model 报错，我们可以由用户决定是否把 Train 当作 Val (仅供代码跑通)
            # 或者在 evaluate_model 里处理空数组。
            # 这里为了代码鲁棒性，稍微 hack 一下：让 Val = Train (Sanity Check)
            # self.val_ids = train_pool 
        else:
            # 原有的逻辑：对于大数据量，正常切分 10%
            n_val = max(1, int(len(train_pool) * 0.1))
            self.val_ids = train_pool[:n_val]
            self.train_ids = train_pool[n_val:]
        self.test_ids = np.array(test_target)
        self.all_exp_ids = all_exp_ids

    def _split_single_product_random(self, all_exp_ids):
        """New logic for standard single-product training."""
        # Just shuffle and split by ratio
        np.random.shuffle(all_exp_ids)
        n_total = len(all_exp_ids)
        
        n_train = int(n_total * self.config.single_prod_train_ratio)
        n_val = int(n_total * self.config.single_prod_val_ratio)
        
        self.train_ids = all_exp_ids[:n_train]
        self.val_ids = all_exp_ids[n_train : n_train + n_val]
        self.test_ids = all_exp_ids[n_train + n_val :]
        self.all_exp_ids = all_exp_ids
    
    def _compute_statistics(self) -> None:
        """Compute normalization statistics from training data only."""
        train_df = self.df[self.df["Unique_ID"].isin(self.train_ids)]
        
        # State statistics
        state_mean = train_df[self.STATE_COLS].mean().values
        state_std = train_df[self.STATE_COLS].std().values + 1e-6
        
        self.state_mean = torch.tensor(
            np.nan_to_num(state_mean, nan=0.0), 
            dtype=torch.float32
        )
        self.state_std = torch.tensor(
            np.nan_to_num(state_std, nan=1.0), 
            dtype=torch.float32
        )
        
        # Input statistics
        self.input_mean = torch.tensor(
            train_df[self.INPUT_COLS].mean().values, 
            dtype=torch.float32
        )
        self.input_std = torch.tensor(
            train_df[self.INPUT_COLS].std().values + 1e-6, 
            dtype=torch.float32
        )
    
    def get_normalization_stats(self) -> NormalizationStats:
        """Get normalization statistics container."""
        return NormalizationStats(
            state_mean=self.state_mean,
            state_std=self.state_std,
            input_mean=self.input_mean,
            input_std=self.input_std,
        )
    
    def get_experiment_data(self, unique_id: str) -> ExperimentData:
        """
        Extract all data for a single experiment.
        
        Args:
            unique_id: Unique experiment identifier.
            
        Returns:
            ExperimentData containing all experiment information.
        """
        exp_df = self.df[self.df["Unique_ID"] == unique_id].sort_values(self.TIME_COL)
        times = exp_df[self.TIME_COL].values.astype(np.float32)
        
        # Get product index
        product_name = exp_df[self.PROD_ID_COL].iloc[0]
        product_idx = self.prod_map[product_name]
        
        # Extract targets and mask
        targets_np = exp_df[self.STATE_COLS].values.astype(np.float32)
        mask_np = ~np.isnan(targets_np)
        targets = torch.tensor(np.nan_to_num(targets_np, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32)
        
        # Initial state with fallback for missing values
        y0 = self._compute_initial_state(targets)
        
        # Build input interpolation functions
        input_funcs = self._build_input_functions(exp_df, times)
        
        # Extract feed events
        events = self._extract_feed_events(exp_df)
        
        initial_vol = float(exp_df.iloc[0][self.VOL_COL])
        
        return ExperimentData(
            times=torch.tensor(times),
            y0=y0,
            targets=targets,
            mask=mask,
            input_funcs=input_funcs,
            events=events,
            initial_vol=initial_vol,
            exp_id=unique_id,
            product_idx=torch.tensor(product_idx, dtype=torch.long),
        )
    
    def _compute_initial_state(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute initial state, handling missing values."""
        y0 = targets[0].clone()
        
        for i in range(len(y0)):
            if torch.isnan(y0[i]):
                valid_values = targets[:, i][~torch.isnan(targets[:, i])]
                y0[i] = valid_values[0] if len(valid_values) > 0 else 0.0
        
        return y0
    
    def _build_input_functions(
        self, 
        exp_df: pd.DataFrame, 
        times: np.ndarray
    ) -> List[Callable]:
        """Build interpolation functions for control inputs."""
        input_funcs = []
        
        for col in self.INPUT_COLS:
            values = exp_df[col].interpolate(method="linear").bfill().ffill().values
            values = np.nan_to_num(values, nan=0.0)
            func = interp1d(times, values, kind="previous", fill_value="extrapolate")
            input_funcs.append(func)
        
        return input_funcs
    
    def _extract_feed_events(self, exp_df: pd.DataFrame) -> List[Dict[str, float]]:
        """Extract feed events from experiment data."""
        events = []
        
        for _, row in exp_df.iterrows():
            glc_mass = float(row[self.FEED_GLC_COL]) if pd.notna(row[self.FEED_GLC_COL]) else 0.0
            gln_mass = float(row[self.FEED_GLN_COL]) if pd.notna(row[self.FEED_GLN_COL]) else 0.0
            vol_add = float(row[self.FEED_VOL_COL]) if pd.notna(row[self.FEED_VOL_COL]) else 0.0
            
            # Only record significant feed events
            if glc_mass > 1e-6 or gln_mass > 1e-6 or vol_add > 1e-6:
                events.append({
                    "time": float(row[self.TIME_COL]),
                    "glc_mass": glc_mass,
                    "gln_mass": gln_mass,
                    "vol_add": vol_add,
                })
        
        events.sort(key=lambda x: x["time"])
        return events


# ==============================================================================
# Neural Network Models
# ==============================================================================

class KineticsNet(nn.Module):
    """
    Neural network for learning kinetic parameters with product embeddings.
    
    Combines normalized state and control inputs with learned product
    embeddings to predict reaction rates.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_products: int, 
        embedding_dim: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize the kinetics network.
        
        Args:
            input_dim: Dimension of physical inputs (state + control).
            output_dim: Number of kinetic rates to predict.
            num_products: Total number of products for embedding.
            embedding_dim: Dimension of product embeddings.
            hidden_dim: Hidden layer dimension.
        """
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
    
    def _initialize_weights(self) -> None:
        """Initialize network weights for stable training."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Small random initialization for embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Small output layer weights for stable initial predictions
        last_layer = self.net[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=1e-4)
        nn.init.zeros_(last_layer.bias)
    
    def forward(self, x: torch.Tensor, product_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Physical inputs [batch_size, input_dim].
            product_idx: Product indices [batch_size] or scalar.
            
        Returns:
            Predicted kinetic rates [batch_size, output_dim].
        """
        emb = self.embedding(product_idx)
        
        # Handle dimension broadcasting
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).expand(x.size(0), -1)
        elif emb.size(0) != x.size(0):
            emb = emb.expand(x.size(0), -1)
        
        combined = torch.cat([x, emb], dim=-1)
        return self.net(combined)

    def expand_embeddings(self, new_num_products: int) -> None:
        """
        Dynamically expand the embedding layer for new products.
        Keeps existing weights and initializes new ones.
        """
        old_emb = self.embedding
        old_num, dim = old_emb.weight.shape
        
        if new_num_products <= old_num:
            return # No expansion needed
            
        print(f"Expanding embedding layer: {old_num} -> {new_num_products}")
        
        # Create new embedding layer
        new_emb = nn.Embedding(new_num_products, dim).to(self.embedding.weight.device)
        
        # 1. Copy old weights (Preserve knowledge)
        with torch.no_grad():
            new_emb.weight[:old_num] = old_emb.weight
            
            # 2. Initialize new weights (e.g., mean of old weights + noise)
            # This is better than random init as it starts in the valid latent space
            mean_emb = old_emb.weight.mean(dim=0)
            for i in range(old_num, new_num_products):
                new_emb.weight[i] = mean_emb + torch.randn_like(mean_emb) * 0.05
        
        self.embedding = new_emb

class ReactionODEFunc(nn.Module):
    """
    ODE function for bioreactor dynamics.
    
    Implements the differential equations for cell growth and metabolism,
    with neural network corrections for unknown kinetics.
    """
    
    def __init__(
        self,
        kinetics_net: KineticsNet,
        input_funcs: List[Callable],
        stats: Dict[str, torch.Tensor],
        product_idx: torch.Tensor,
    ):
        """
        Initialize ODE function.
        
        Args:
            kinetics_net: Neural network for kinetic rate prediction.
            input_funcs: Interpolation functions for control inputs.
            stats: Normalization statistics dictionary.
            product_idx: Product index for this experiment.
        """
        super().__init__()
        self.net = kinetics_net
        self.input_funcs = input_funcs
        self.product_idx = product_idx
        
        self.register_buffer("state_mean", stats["state_mean"])
        self.register_buffer("state_std", stats["state_std"])
        self.register_buffer("input_mean", stats["input_mean"])
        self.register_buffer("input_std", stats["input_std"])
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivatives at time t.
        
        Args:
            t: Current time.
            y: Current state [VCD, Glc, Gln, Amm, Lac, product].
            
        Returns:
            State derivatives.
        """
        # Get control inputs at current time
        t_np = t.item()
        controls = self._get_controls(t_np, y.device)
        
        # Normalize inputs
        norm_y = (y - self.state_mean) / self.state_std
        norm_controls = (controls - self.input_mean) / self.input_std
        
        metabolites_indices = [0, 1, 2, 3, 4] # exclude titer
        norm_y_features = norm_y[metabolites_indices]
        nn_input = torch.cat([norm_y_features, norm_controls], dim=-1)
        
        # Predict kinetic rates
        rates = self.net(nn_input.unsqueeze(0), self.product_idx).squeeze(0)
        
        # # Unpack rates
        mu, q_glc, q_gln, q_amm, q_lac, q_prod = (
            rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]
        )
        x_v = torch.nn.functional.softplus(y[0]) 
        
        # Compute derivatives
        d_vcd = mu * x_v
        d_glc = -q_glc * x_v
        d_gln = -q_gln * x_v
        d_amm = q_amm * x_v
        d_lac = q_lac * x_v
        d_prod = q_prod * x_v
        
        return torch.stack([d_vcd, d_glc, d_gln, d_amm, d_lac, d_prod])
    
    def _get_controls(self, t: float, device: torch.device) -> torch.Tensor:
        """Get control inputs at specified time."""
        controls_np = np.array(
            [f(t) for f in self.input_funcs], 
            dtype=np.float32
        )
        return torch.from_numpy(controls_np).to(device)


# ==============================================================================
# ODE Solver with Event Handling
# ==============================================================================

def solve_with_events(
    net: KineticsNet,
    exp_data: ExperimentData,
    stats: Dict[str, torch.Tensor],
    t_eval_points: Optional[torch.Tensor] = None,
    config: Config = CONFIG,
) -> torch.Tensor:
    """
    Solve ODE system with discrete feed events.
    
    Handles piecewise integration with concentration updates at feed times.
    
    Args:
        net: Kinetics neural network.
        exp_data: Experiment data container.
        stats: Normalization statistics.
        t_eval_points: Optional evaluation time points.
        config: Configuration object.
        
    Returns:
        Predicted states at evaluation times.
    """
    device = config.device
    y_current = exp_data.y0.to(device)
    current_vol = exp_data.initial_vol
    product_idx = exp_data.product_idx.to(device)
    
    # Determine evaluation times
    if t_eval_points is None:
        t_targets = exp_data.times.cpu().numpy()
    else:
        t_targets = t_eval_points.cpu().numpy()
    
    # Merge evaluation times with feed event times
    feed_times = [e["time"] for e in exp_data.events]
    all_stops = sorted(set(np.concatenate([t_targets, feed_times])))
    
    # Store predictions at target times
    pred_dict = {}
    if abs(all_stops[0] - t_targets[0]) < 1e-4:
        pred_dict[all_stops[0]] = y_current.clone()
    
    # Initialize ODE function
    ode_func = ReactionODEFunc(net, exp_data.input_funcs, stats, product_idx)
    ode_func = ode_func.to(device)
    
    current_t = all_stops[0]
    
    # Integrate between stop points
    for next_t in all_stops[1:]:
        if next_t <= current_t:
            continue
        
        t_span = torch.tensor([current_t, next_t], dtype=torch.float32).to(device)
        
        sol = odeint(
            ode_func, 
            y_current, 
            t_span,
            method=config.ode_method,
            rtol=config.ode_rtol,
            atol=config.ode_atol,
        )
        y_next = sol[-1]
        
        # Apply feed events at this time
        y_next, current_vol = _apply_feed_events(
            y_next, 
            current_vol, 
            exp_data.events, 
            next_t
        )
        
        # Store prediction if this is a target time
        if np.any(np.isclose(t_targets, next_t, atol=1e-4)):
            pred_dict[next_t] = y_next.clone()
        
        current_t = next_t
        y_current = y_next
    
    # Collect predictions in order
    final_preds = []
    for t in t_targets:
        closest_t = min(pred_dict.keys(), key=lambda x: abs(x - t))
        final_preds.append(pred_dict[closest_t])
    
    return torch.stack(final_preds)


def _apply_feed_events(
    y: torch.Tensor,
    current_vol: float,
    events: List[Dict[str, float]],
    current_time: float,
) -> Tuple[torch.Tensor, float]:
    """
    Apply feed events at the current time.
    
    Updates concentrations based on mass addition and volume change.
    
    Args:
        y: Current state vector.
        current_vol: Current reactor volume.
        events: List of all feed events.
        current_time: Current simulation time.
        
    Returns:
        Updated state and volume.
    """
    events_now = [e for e in events if abs(e["time"] - current_time) < 1e-4]
    
    if not events_now:
        return y, current_vol
    
    # Convert to total masses
    vcd, glc, gln, amm, lac, prod = y
    total_vcd = vcd * current_vol
    total_glc = glc * current_vol
    total_gln = gln * current_vol
    total_amm = amm * current_vol
    total_lac = lac * current_vol
    total_prod = prod * current_vol
    
    # Apply each feed event
    for event in events_now:
        total_glc += event["glc_mass"]
        total_gln += event["gln_mass"]
        current_vol += event["vol_add"]
    
    # Convert back to concentrations
    y_updated = torch.stack([
        total_vcd / current_vol,
        total_glc / current_vol,
        total_gln / current_vol,
        total_amm / current_vol,
        total_lac / current_vol,
        total_prod / current_vol,
    ])
    
    return y_updated, current_vol


# ==============================================================================
# Loss Functions and Training Utilities
# ==============================================================================

def masked_relative_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std_vec: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute masked relative mean squared error.
    
    Normalizes MSE by standard deviation for each state variable.
    
    Args:
        pred: Predicted values.
        target: Target values.
        mask: Binary mask for valid values.
        std_vec: Standard deviation for normalization.
        eps: Small constant for numerical stability.
        
    Returns:
        Relative RMSE per state variable.
    """
    diff = (pred - target) * mask
    n_effective = torch.clamp(mask.sum(dim=0), min=1.0)
    mse = (diff ** 2).sum(dim=0) / n_effective
    rmse = torch.sqrt(mse)
    std = torch.clamp(std_vec, min=eps)
    return rmse / std


class EarlyStopping:
    """Early stopping handler with model checkpointing."""
    
    def __init__(
        self,
        patience: int = 20,
        delta: float = 0.0,
        save_path: str = "best_model.pth",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            delta: Minimum change to qualify as improvement.
            save_path: Path to save best model.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
    
    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        stats: Optional[Dict] = None,
        product_map: Optional[Dict] = {},
    ) -> None:
        """Check for improvement and update state."""
        if np.isnan(val_loss):
            return
        
        score = -val_loss
        
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(model, stats, product_map)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _save_checkpoint(self, model: nn.Module, stats: Optional[Dict], product_map: Optional[Dict]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "product_map": product_map
        }
        torch.save(checkpoint, self.save_path)


# ==============================================================================
# Evaluation and Visualization
# ==============================================================================

def evaluate_model(
    net: KineticsNet,
    dataset: BioreactorDataset,
    stats: Dict[str, torch.Tensor],
    exp_ids: np.ndarray,
    config: Config = CONFIG,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate model on a set of experiments.
    
    Args:
        net: Trained kinetics network.
        dataset: Dataset object.
        stats: Normalization statistics.
        exp_ids: Experiment IDs to evaluate.
        config: Configuration object.
        
    Returns:
        Average NRMSE and per-variable NRMSE array.
    """
    if len(exp_ids) == 0:
        return 0.0, np.zeros(6)
    
    net.eval()
    total_nrmse = torch.zeros(6).to(config.device)
    count = 0
    
    with torch.no_grad():
        for exp_id in exp_ids:
            exp_data = dataset.get_experiment_data(exp_id)
            targets = exp_data.targets.to(config.device)
            mask = exp_data.mask.to(config.device)
            
            try:
                pred_y = solve_with_events(net, exp_data, stats, config=config)
                
                if torch.isnan(pred_y).any():
                    continue
                
                nrmse_vec = masked_relative_mse(
                    pred_y, targets, mask, stats["state_std"]
                )
                total_nrmse += nrmse_vec
                count += 1
                
            except Exception as e:
                print(f"Error evaluating {exp_id}: {e}")
    
    avg_nrmse = total_nrmse / max(count, 1)
    return avg_nrmse.mean().item(), avg_nrmse.cpu().numpy()


def plot_product_embeddings(
    net: KineticsNet,
    dataset: BioreactorDataset,
    save_path: str = "embedding_space.png",
    config: Config = CONFIG,
) -> None:
    """
    Visualize learned product embeddings in 2D space.
    
    Args:
        net: Trained kinetics network.
        dataset: Dataset object.
        save_path: Path to save the plot.
        config: Configuration object.
    """
    if dataset.num_products < 2:
        print("Single product detected. Skipping embedding plot.")
        return

    net.eval()
    
    if config.embedding_dim != 2:
        print("Warning: Only 2D embeddings supported for direct plotting.")
        return
    
    embeddings = net.embedding.weight.data.cpu().numpy()
    product_names = [
        name for name, _ in sorted(dataset.prod_map.items(), key=lambda x: x[1])
    ]
    
    plt.figure(figsize=(8, 6))
    
    # Plot all products
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c="blue", s=100, alpha=0.7)
    
    # Add labels
    for i, name in enumerate(product_names):
        is_target = name == config.target_product
        color = "red" if is_target else "black"
        weight = "bold" if is_target else "normal"
        
        plt.text(
            embeddings[i, 0] + 0.02,
            embeddings[i, 1] + 0.02,
            name,
            fontsize=12,
            color=color,
            fontweight=weight,
        )
        
        if is_target:
            plt.scatter(
                embeddings[i, 0],
                embeddings[i, 1],
                c="red",
                s=150,
                marker="*",
                label="Target (New)",
            )
    
    plt.title(
        f"Learned Product Embeddings (2D)\nTarget: {config.target_product}",
        fontsize=14,
    )
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Embedding plot saved to {save_path}")


def visualize_prediction(
    net: KineticsNet,
    dataset: BioreactorDataset,
    stats: Dict[str, torch.Tensor],
    exp_id: str,
    save_path: Optional[str] = None,
    config: Config = CONFIG,
) -> None:
    """
    Visualize model predictions vs experimental data.
    
    Args:
        net: Trained kinetics network.
        dataset: Dataset object.
        stats: Normalization statistics.
        exp_id: Experiment ID to visualize.
        save_path: Optional path to save the plot.
        config: Configuration object.
    """
    net.eval()
    exp_data = dataset.get_experiment_data(exp_id)
    
    t_start = exp_data.times[0].item()
    t_end = exp_data.times[-1].item()
    t_dense = torch.linspace(t_start, t_end, 200).to(config.device)
    
    with torch.no_grad():
        pred_y = solve_with_events(net, exp_data, stats, t_eval_points=t_dense, config=config)
    
    t_pred = t_dense.cpu().numpy()
    y_pred = pred_y.cpu().numpy()
    t_true = exp_data.times.cpu().numpy()
    y_true = exp_data.targets.cpu().numpy()
    mask = exp_data.mask.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(dataset.STATE_COLS):
        ax = axes[i]
        ax.plot(t_pred, y_pred[:, i], label="Model", color="tab:blue")
        
        # Only plot valid experimental points
        valid_mask = mask[:, i].astype(bool)
        ax.scatter(
            t_true[valid_mask],
            y_true[valid_mask, i],
            label="Experimental",
            color="tab:red",
            zorder=5,
        )
        
        ax.set_title(col)
        ax.set_xlabel("Time [h]")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        if i == 0:
            ax.legend()
    
    plt.suptitle(f"Experiment: {exp_id}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

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

# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_model(
    file_path: str,
    config: Config = CONFIG,
) -> Tuple[KineticsNet, BioreactorDataset, Dict[str, torch.Tensor]]:
    """
    Train the multi-task kinetics model.
    
    Args:
        file_path: Path to training data CSV.
        config: Configuration object.
        
    Returns:
        Trained model, dataset, and normalization statistics.
    """
    # Load and prepare data
    dataset = BioreactorDataset(file_path, config)
    stats = dataset.get_normalization_stats().to_device(config.device)
    
    # Initialize model
    net = KineticsNet(
        input_dim=8,
        output_dim=6,
        num_products=dataset.num_products,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
    )
    early_stopping = EarlyStopping(
        patience=config.patience,
        save_path=config.model_save_path,
    )
    
    print(f"\n--- Starting Training (Max Epochs: {config.epochs}) ---")
    history = {"train_loss": [], "val_nrmse_avg": [], "val_nrmse_vars": []}
    accumulation_steps = config.batch_size
    
    for epoch in range(config.epochs):
        net.train()
        batch_loss = 0.0
        optimizer.zero_grad()
        
        # Shuffle training data
        epoch_train_ids = np.random.permutation(dataset.train_ids)
        
        for i, exp_id in enumerate(epoch_train_ids):
            exp_data = dataset.get_experiment_data(exp_id)
            targets = exp_data.targets.to(config.device)
            mask = exp_data.mask.to(config.device)
            
            # Forward pass
            try:
                pred_y = solve_with_events(net, exp_data, stats, config=config)
            except Exception:
                continue
            
            # Compute loss
            nrmse_vec = masked_relative_mse(pred_y, targets, mask, stats["state_std"])
            loss = torch.mean(nrmse_vec)
            
            if torch.isnan(loss):
                continue
            
            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(epoch_train_ids):
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            batch_loss += loss.item() * accumulation_steps
        
        avg_loss = batch_loss / len(epoch_train_ids)
        history["train_loss"].append(avg_loss)
        
        # Validation
        if (epoch + 1) % config.print_freq == 0:
            val_avg, val_vec = evaluate_model(net, dataset, stats, dataset.val_ids, config)
            history["val_nrmse_avg"].append(val_avg)
            history["val_nrmse_vars"].append(val_vec)
            print(
                f"Epoch {epoch + 1:04d} | "
                f"Loss: {avg_loss:.5f} | "
                f"Val Avg: {val_avg:.2f} | "
                f"Prod Err: {val_vec[5]:.2f}"
            )
            
            scheduler.step(val_avg)
            early_stopping(val_avg, net, stats=stats, product_map=dataset.prod_map)
            
            if early_stopping.early_stop:
                print("--- Early stopping triggered ---")
                break
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load(early_stopping.save_path, map_location=config.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    
    return net, dataset, stats, history


# ==============================================================================
# Task Manager
# ==============================================================================

def run_task(
    task_type: str = "train",
    data_path: str = "data/real_world_data.csv",
    model_path: str = "best_mtl_model.pth",
    config: Config = CONFIG,
) -> None:
    """
    Execute training, evaluation, or simulation task.
    
    Args:
        task_type: One of "train", "evaluate", or "simulation".
        data_path: Path to data file.
        model_path: Path to model checkpoint.
        config: Configuration object.
    """
    device = config.device
    
    if task_type == "train":
        _run_training_task(data_path, config)
        
    elif task_type == "evaluate":
        _run_evaluation_task(data_path, model_path, config)
        
    elif task_type == "simulation":
        _run_simulation_task(data_path, model_path, config)
        
    else:
        print(f"Unknown task type: {task_type}")


def _run_training_task(data_path: str, config: Config) -> None:
    """Execute training task with evaluation and visualization."""
    print(f"\n>>> Multi-Task Training: Target {config.target_product}")
    
    model, dataset, stats, history = train_model(data_path, config)
    plot_training_metrics(history, dataset.STATE_COLS)
    
    # Evaluate on test set
    print("\n--- Evaluation on Unseen Target Runs ---")
    test_avg, test_vec = evaluate_model(model, dataset, stats, dataset.test_ids, config)
    
    print(f"Test NRMSE (Avg): {test_avg:.2f}")
    for i, col in enumerate(dataset.STATE_COLS):
        print(f"  {col}: {test_vec[i]:.2f}")
    
    # Visualize predictions
    for exp_id in dataset.test_ids[:3]:
        visualize_prediction(
            model, dataset, stats, exp_id,
            save_path=f"test_pred_{exp_id}.png",
            config=config,
        )
    
    # Visualize embeddings
    plot_product_embeddings(model, dataset, config=config)


def fine_tune_task(
    hist_model_path: str,
    novel_data_path: str,
    n_shots: int = 2,
    config: Config = CONFIG,
) -> None:
    """
    Fine-tune a pre-trained model on a new product.
    """
    print(f"\n>>> Starting Fine-Tuning Task")
    print(f"Base Model: {hist_model_path}")
    print(f"New Data: {novel_data_path}")
    
    # 1. Load Base Model & Stats
    if not os.path.exists(hist_model_path):
        raise FileNotFoundError("Base model not found!")
    
    checkpoint = torch.load(hist_model_path, map_location=config.device)
    base_stats = checkpoint["stats"]
    base_prod_map = checkpoint["product_map"]
    
    # 2. Load New Dataset (Inheriting Map)
    ft_config = copy.deepcopy(config)
    ft_config.n_target_shots = n_shots
    
    dataset = BioreactorDataset(novel_data_path, ft_config, existing_prod_map=base_prod_map)
    
    stats = base_stats 
    stats_tensor = {k: v.to(config.device) for k, v in stats.items()}
    
    # 3. Initialize Model Structure (Old Size)
    # Retrieve old number of products from checkpoint shape
    old_num_products = checkpoint["model_state_dict"]["embedding.weight"].shape[0]
    
    net = KineticsNet(
        input_dim=8, 
        output_dim=6,
        num_products=old_num_products, # Start with old size
        embedding_dim=config.embedding_dim
    ).to(config.device)
    
    # 4. Load Weights
    net.load_state_dict(checkpoint["model_state_dict"])
    
    # 5. Expand Model for New Product
    if dataset.num_products > old_num_products:
        net.expand_embeddings(dataset.num_products)
    
    # 6. Fine-Tuning Setup
    # Strategy: Train Embedding + MLP with small LR
    # Alternatively: Freeze MLP, train only Embedding (requires_grad=False)
    
    # For robust 2-shot, we usually use a smaller LR
    ft_lr = config.learning_rate * 0.5 
    
    optimizer = optim.Adam(net.parameters(), lr=ft_lr, weight_decay=1e-3)
    
    # 7. Training Loop (Only on new data)
    print(f"Fine-tuning on {len(dataset.train_ids)} shots...")
    net.train()
    
    # Fine-tune epochs can be smaller or derived from config
    ft_epochs = 200 
    
    for epoch in range(ft_epochs):
        batch_loss = 0.0
        optimizer.zero_grad()
        
        # Shuffle IDs
        ids = np.random.permutation(dataset.train_ids)
        
        for exp_id in ids:
            exp_data = dataset.get_experiment_data(exp_id)
            targets = exp_data.targets.to(config.device)
            mask = exp_data.mask.to(config.device)
            
            try:
                pred_y = solve_with_events(net, exp_data, stats_tensor, config=config)
                loss = torch.mean(masked_relative_mse(pred_y, targets, mask, stats_tensor["state_std"]))
                
                loss.backward()
                batch_loss += loss.item()
            except Exception as e:
                print(f"FT Error: {e}")
                continue
        
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"FT Epoch {epoch+1} | Loss: {batch_loss/len(ids):.5f}")

    plot_product_embeddings(net, dataset, save_path="finetune_embedding.png", config=config)

    # 8. Evaluation
    print("\n--- Evaluating on Remaining New Data ---")
    test_avg, test_vec = evaluate_model(net, dataset, stats_tensor, dataset.test_ids, config)
    print(f"Test NRMSE (New Product): {test_avg:.2f}")

    # Visualize
    if len(dataset.test_ids) > 0:
        for i in range(len(dataset.test_ids)):
            visualize_prediction(net, dataset, stats_tensor, dataset.test_ids[i], save_path=f"ft_result_{i}.png", config=config)
        
    return net

def _run_evaluation_task(data_path: str, model_path: str, config: Config) -> None:
    """Execute evaluation task on saved model."""
    print(f"\n>>> Starting Task: EVALUATION using {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found!")
        return
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device)
    stats = checkpoint["stats"]
    
    # Load dataset
    dataset = BioreactorDataset(data_path, config)
    
    # Initialize and load model
    net = KineticsNet(
        input_dim=8,
        output_dim=6,
        num_products=dataset.num_products,
        embedding_dim=config.embedding_dim,
    ).to(config.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate
    test_avg, test_vec = evaluate_model(net, dataset, stats, dataset.test_ids, config)
    
    print(f"NRMSE on {data_path} (Test Set): {test_avg:.2f}")
    for i, col in enumerate(dataset.STATE_COLS):
        print(f"  {col}: {test_vec[i]:.2f}")
    
    # Visualize samples
    if len(dataset.test_ids) > 0:
        for exp_id in dataset.test_ids[:3]:
            visualize_prediction(
                net, dataset, stats, exp_id,
                save_path=f"eval_{exp_id}.png",
                config=config,
            )
    
    plot_product_embeddings(net, dataset, save_path="eval_embedding.png", config=config)


def _run_simulation_task(data_path: str, model_path: str, config: Config) -> None:
    """Execute simulation task on new data."""
    print(f"\n>>> Starting Task: SIMULATION using {data_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found!")
        return
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device)
    train_stats = checkpoint["stats"]
    
    # Load simulation dataset
    sim_dataset = BioreactorDataset(data_path, config)
    
    # Initialize and load model
    net = KineticsNet(
        input_dim=8,
        output_dim=6,
        num_products=sim_dataset.num_products,
        embedding_dim=config.embedding_dim,
    ).to(config.device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    
    # Run simulations
    print(f"Simulating {len(sim_dataset.all_exp_ids)} experiments...")
    
    for exp_id in sim_dataset.all_exp_ids:
        visualize_prediction(
            net, sim_dataset, train_stats, exp_id,
            save_path=f"sim_{exp_id}.png",
            config=config,
        )
    
    print("Simulation complete.")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":

    # Step 2: 只有在有了 base_model.pth 后才运行这个
    if os.path.exists("base_model.pth") and os.path.exists("data/novel.csv"):
        # 为微调创建一个新的配置（通常需要更小的学习率）
        ft_config = copy.deepcopy(CONFIG)
        ft_config.learning_rate = 0.001 
        ft_config.target_product = "NP"  # <--- 修正这里
        fine_tune_task(
            hist_model_path="base_model.pth",
            novel_data_path="data/novel.csv",
            n_shots=4,  # 2-shot 微调
            config=ft_config
        )
    # else:
    #     # Step 1: 训练 Base Model
    #     DATA_FILE = "data/hist.csv"
    #     if os.path.exists(DATA_FILE):
    #          CONFIG.model_save_path = "base_model.pth"
    #          CONFIG.target_product = "NON_EXISTENT_PRODUCT"
    #          CONFIG.n_target_shots = 0
    #          run_task(task_type="train", data_path=DATA_FILE)