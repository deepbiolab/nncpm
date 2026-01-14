import pandas as pd
import numpy as np
import os

def mock_missing_values(
    input_file='novel.csv', 
    output_file='real_world_data.csv',
    scenarios=None
):
    """
    读取完整数据集，根据设定的场景引入缺失值 (NaN)，模拟真实实验数据。
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    original_rows = len(df)
    
    # 复制一份用于修改
    mock_df = df.copy()

    # 默认场景配置 (如果未提供)
    if scenarios is None:
        scenarios = {
            # 场景1: Titer (product) 只在最后 3 天测量
            'late_stage': {
                'columns': ['product'],
                'days_to_keep': 3.0  # 保留最后 3 天 (72小时)
            },
            # 场景2: 代谢物 (Glc, Lac, Amm) 隔天一测 (每 48h)
            'sparse_sampling': {
                'columns': ['Glc', 'Lac', 'Amm', 'Gln'],
                'interval_hours': 48.0,
                'keep_start': True # 是否强制保留 t=0 的点
            },
            # 场景3: VCD 通常测得比较勤，假设它是 24h 一测 (保持原样或模拟随机丢失)
            'random_drop': {
                'columns': ['VCD'],
                'drop_prob': 0.05 # 5% 的概率随机丢失数据 (模拟仪器故障)
            }
        }

    # 按实验批次处理 (Exp_ID)
    grouped = mock_df.groupby('Exp_ID')
    
    # 收集处理后的索引列表
    indices_to_mask = {col: [] for strategy in scenarios.values() for col in strategy['columns']}

    print("\nApplying missing value strategies...")

    for exp_id, group in grouped:
        # 获取该批次的最大时间 (收获时间)
        max_time = group['time[h]'].max()
        
        # --- 策略 A: Late Stage Only (后期测量) ---
        if 'late_stage' in scenarios:
            cfg = scenarios['late_stage']
            cutoff_time = max_time - (cfg['days_to_keep'] * 24.0)
            
            for col in cfg['columns']:
                # 找到所有时间小于 cutoff 的行的索引
                mask_idx = group[group['time[h]'] < cutoff_time].index
                mock_df.loc[mask_idx, col] = np.nan

        # --- 策略 B: Sparse Sampling (稀疏采样) ---
        if 'sparse_sampling' in scenarios:
            cfg = scenarios['sparse_sampling']
            interval = cfg['interval_hours']
            
            for col in cfg['columns']:
                # 逻辑: 如果 (time % interval) != 0，则设为 NaN
                # 注意浮点数取模需要容差
                is_interval_point = (np.abs(group['time[h]'] % interval) < 1e-4)
                
                if cfg.get('keep_start', True):
                    # 强制保留 t=0
                    is_start = (group['time[h]'] == 0)
                    keep_mask = is_interval_point | is_start
                else:
                    keep_mask = is_interval_point
                
                # 将不需要保留的行设为 NaN
                mask_idx = group[~keep_mask].index
                mock_df.loc[mask_idx, col] = np.nan

        # --- 策略 C: Random Drop (随机丢失) ---
        if 'random_drop' in scenarios:
            cfg = scenarios['random_drop']
            prob = cfg['drop_prob']
            
            for col in cfg['columns']:
                # 随机生成掩码
                # 0: keep, 1: drop
                rand_mask = np.random.rand(len(group)) < prob
                
                # 即使是随机丢失，通常也要保留 t=0 (作为初始条件)
                if group['time[h]'].min() == 0:
                    start_idx_in_group = (group['time[h]'] == 0).values
                    rand_mask[start_idx_in_group] = False # 不要 drop t=0
                
                # 获取全局索引
                global_indices = group.index[rand_mask]
                mock_df.loc[global_indices, col] = np.nan

    # --- 统计并保存 ---
    print("\nMissing Data Summary:")
    print("-" * 30)
    for col in mock_df.columns:
        if col in ['Exp_ID', 'Product_ID']: continue
        n_missing = mock_df[col].isna().sum()
        pct_missing = (n_missing / original_rows) * 100
        if n_missing > 0:
            print(f"{col:<10}: {n_missing} missing ({pct_missing:.1f}%)")
    
    # 关键检查：确保 Cmd_ (控制变量) 和 Time 没有被修改
    # 真实场景中，控制变量通常是机器记录的，不会缺失
    print("-" * 30)
    print("Verifying critical columns (Should be 0 missing):")
    critical_cols = [c for c in mock_df.columns if 'Cmd_' in c or 'time' in c or 'Volume' in c]
    for c in critical_cols:
        miss = mock_df[c].isna().sum()
        print(f"{c:<20}: {miss}")

    mock_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully generated '{output_file}' with realistic missing patterns.")

    # 打印前20行预览
    print("\nPreview of generated data (Head 20):")
    print(mock_df[['time[h]', 'VCD', 'Glc', 'product', 'Exp_ID']].head(20))

if __name__ == "__main__":
    # 配置你的模拟策略
    my_scenarios = {
        # 1. 产物 (Titer): 极度稀疏，只在最后 2 天测量 (例如 Day 13, 14)
        'late_stage': {
            'columns': ['product'],
            'days_to_keep': 3.0 
        },
        
        # 2. 代谢物 (Glc/Gln): 隔天测量 (0, 48, 96...)
        # 'sparse_sampling': {
        #     'columns': ['Glc', 'Gln', 'Amm', 'Lac'],
        #     'interval_hours': 48.0,
        #     'keep_start': True
        # },
        
        # 3. 细胞密度 (VCD): 每天测量 (保持原样，或少量随机丢失)
        # 这里我们假设 VCD 每天都测，不做处理，或者你可以取消注释下面这行：
        # 'random_drop': {'columns': ['VCD'], 'drop_prob': 0.05}
    }

    # 运行
    # 假设输入文件是你在当前目录下的 test.csv (或 novel.csv)
    mock_missing_values(input_file='data/test.csv', output_file='data/real_world_data.csv', scenarios=my_scenarios)