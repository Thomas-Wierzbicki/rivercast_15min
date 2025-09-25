import pandas as pd
import yaml
from pathlib import Path

CFG = yaml.safe_load(open('config/config.yaml'))

RAW = Path('data/raw')
OUT = Path('data/interim')
OUT.mkdir(parents=True, exist_ok=True)

def load_raw(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=['ts'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # sort & enforce raster using groupby-asfreq
    df = df.sort_values(['station_id','ts'])
    def to_raster(g):
        g = g.set_index('ts').asfreq(CFG['raster'])
        return g
    df = df.groupby('station_id', group_keys=False).apply(to_raster).reset_index()
    return df

def merge_exogenous(df):
    exo_cfg = CFG.get('features_exogenous', {})
    for name, opt in exo_cfg.items():
        if not opt.get('enabled', False): 
            continue
        p = Path(opt['path'])
        if not p.exists():
            print(f"[WARN] Exogenous file not found: {p}")
            continue
        x = pd.read_csv(p, parse_dates=['ts'])
        df = df.merge(x, on=['ts','station_id'], how='left')
    return df

if __name__ == "__main__":
    # base series (must include at least 'ts','station_id' and either 'q_cms' or 'h_cm')
    base_paths = [p for p in RAW.glob('*.csv') if p.name != 'icon_forecast.csv' and p.name != 'soil_moisture.csv']
    if not base_paths:
        raise SystemExit("No CSV files found in data/raw. Add raw data CSVs first.")
    df = load_raw(base_paths)
    df = merge_exogenous(df)
    df.to_parquet(OUT/'all.parquet')
    print(f"Loaded & merged: {len(df)} rows -> {OUT/'all.parquet'}")
