import pandas as pd
import numpy as np
from pathlib import Path

INP = Path('data/interim/all.parquet')
OUT = Path('data/processed')
OUT.mkdir(parents=True, exist_ok=True)

def qc_group(g, max_gap=8):
    # hard plausibility
    for c in ['q_cms','rain_mm','temp_c']:
        if c in g.columns:
            g.loc[g[c] < 0, c] = np.nan
    # short gap filling (<= max_gap)
    g = g.set_index('ts')
    for c in ['q_cms','rain_mm','temp_c']:
        if c in g.columns:
            g[c] = g[c].interpolate(limit=max_gap)
    return g.reset_index()

if __name__ == "__main__":
    df = pd.read_parquet(INP)
    df = (df.groupby('station_id', group_keys=False)
            .apply(qc_group)
            .dropna(subset=['q_cms']))
    df.to_parquet(OUT/'clean.parquet')
    print(f"QC done -> {OUT/'clean.parquet'}")
