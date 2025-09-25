# src/eval/hindcast.py
"""
Hindcast für den 'per_station'-Modus:
Erzeugt historische p10/p50/p90-Zeitreihen je Station & Horizont, um Fan-Charts zu plotten.

Beispiel:
  python -m src.eval.hindcast --horizons 24 48 96 --out data/processed/hindcast

Voraussetzungen:
- config/config.yaml: training_mode: 'per_station'
- train_baseline.py wurde ausgeführt (Modelle unter artifacts/{station}/{H}/)
- data/processed/feat.parquet existiert
"""

import argparse, json
from pathlib import Path
import pandas as pd
import joblib
import yaml

CFG = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
ART = Path('artifacts')

def load_features_df() -> pd.DataFrame:
    df = pd.read_parquet('data/processed/feat.parquet').sort_values(['station_id','ts'])
    return df

def run_hindcast(df: pd.DataFrame, horizons, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    stations = df['station_id'].dropna().unique().tolist()
    written = []

    for sid in stations:
        g = df[df['station_id'] == sid].copy()
        for H in horizons:
            mdir = ART / sid / str(H)
            if not mdir.exists():
                print(f"[SKIP] {sid} H={H}: kein Modellordner {mdir}")
                continue

            meta = json.loads((mdir/'meta.json').read_text(encoding='utf-8'))
            feats = meta['features']
            m10 = joblib.load(mdir/'model_p10.lgb')
            m50 = joblib.load(mdir/'model_p50.lgb')
            m90 = joblib.load(mdir/'model_p90.lgb')

            X = g[feats].copy()
            g['y_true'] = g[CFG.get('target_col','q_cms')].shift(-H)

            p10 = m10.predict(X)
            p50 = m50.predict(X)
            p90 = m90.predict(X)

            out = pd.DataFrame({
                'ts': g['ts'],
                'station_id': sid,
                'horizon_steps': H,
                'horizon_minutes': H * 15,
                'p10': p10,
                'p50': p50,
                'p90': p90,
                'y_true': g['y_true']
            }).dropna(subset=['y_true'])

            fout = outdir / f"hindcast_{sid}_{H}.parquet"
            out.to_parquet(fout)
            written.append(str(fout))
            print(f"[OK] {sid} H={H} -> {fout}")
    return written

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=None, help="Horizonte in 15-min Schritten, z. B. 24 48 96")
    ap.add_argument("--out", type=str, default="data/processed/hindcast", help="Ausgabeordner")
    args = ap.parse_args()

    horizons = args.horizons or (CFG.get('horizon_steps_list') or [24, 48, 96])
    outdir = Path(args.out)

    df = load_features_df()
    run_hindcast(df, horizons, outdir)
