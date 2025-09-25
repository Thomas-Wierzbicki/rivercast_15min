import pandas as pd
from pathlib import Path
import yaml

CFG = yaml.safe_load(open('config/config.yaml'))

INP = Path('data/processed/clean.parquet')
OUT = Path('data/processed/feat.parquet')

TARGET = CFG.get('target_col', 'q_cms')

def make_features(g, lags=(1,2,4,8,12,24,48), rolls=(4,8,24)):
    g = g.sort_values('ts').copy()
    # Basic lags/rolls for target and common exogenous
    for L in lags:
        if TARGET in g: g[f'{TARGET}_lag{L}'] = g[TARGET].shift(L)
        if 'rain_mm' in g: g[f'rain_lag{L}'] = g['rain_mm'].shift(L)
        if 'icon_rr_mm' in g: g[f'icon_rr_lag{L}'] = g['icon_rr_mm'].shift(L)
        if 'sm_pct' in g: g[f'sm_lag{L}'] = g['sm_pct'].shift(L)
    for R in rolls:
        if TARGET in g: g[f'{TARGET}_roll{R}'] = g[TARGET].rolling(R, min_periods=1).mean()
        if 'rain_mm' in g: g[f'rain_roll{R}'] = g['rain_mm'].rolling(R, min_periods=1).sum()
        if 'icon_rr_mm' in g: g[f'icon_rr_roll{R}'] = g['icon_rr_mm'].rolling(R, min_periods=1).sum()
        if 'sm_pct' in g: g[f'sm_roll{R}'] = g['sm_pct'].rolling(R, min_periods=1).mean()
    g['doy'] = g['ts'].dt.dayofyear
    g['hod'] = g['ts'].dt.hour
    return g

if __name__ == "__main__":
    df = pd.read_parquet(INP)
    feat = df.groupby('station_id', group_keys=False).apply(make_features)
    # shift target by each horizon later during training/serving
    feat.to_parquet(OUT)
    print(f"Features built -> {OUT}")
