import pandas as pd
import numpy as np
import joblib, json, yaml
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from src.utils.metrics import nse, kge
from src.utils.metrics import nse, kge, rmse as rmse_fn

CFG = yaml.safe_load(open('config/config.yaml'))
INP = Path('data/processed/feat.parquet')
ART = Path('artifacts')
ART.mkdir(parents=True, exist_ok=True)

TARGET = CFG.get('target_col','q_cms')
H_LIST = CFG.get('horizon_steps_list', [24, 48, 96])  # 6h, 12h, 24h at 15-min

def qmodel(alpha):
    return LGBMRegressor(objective='quantile', alpha=alpha,
                         n_estimators=700, learning_rate=0.03,
                         num_leaves=96, subsample=0.8, colsample_bytree=0.8,
                         random_state=42)

if __name__ == "__main__":
    df = pd.read_parquet(INP).sort_values(['station_id','ts'])
    base_exclude = ['ts','station_id','q_cms','h_cm']
    features = [c for c in df.columns if c not in base_exclude]
    # Train per horizon separate models (p10/p50/p90)
    report = {}
    for H in H_LIST:
        d = df.copy()
        d['y'] = d.groupby('station_id')[TARGET].shift(-H)
        d = d.dropna(subset=['y'])
        X = d[features]; y = d['y']
        if len(d) < 300:
            print(f"[WARN] Few samples for horizon {H}")
        tscv = TimeSeriesSplit(n_splits=5)
        preds, trues = [], []
        for tr, te in tscv.split(X):
            m = qmodel(0.5).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            preds.append(p); trues.append(y.iloc[te].values)
        pred = np.concatenate(preds); true = np.concatenate(trues)
        mae = mean_absolute_error(true, pred)

        rmse_val = rmse_fn(true, pred)

        res = {
            "MAE": float(mae),
            "RMSE": float(rmse_val),
            "NSE": float(nse(true, pred)),
            "KGE": float(kge(true, pred))
        }




        report[str(H)] = res

        print(f"H={H}  MAE: {mae:.4f}  RMSE: {rmse_val:.4f}  NSE: {nse(true, pred):.4f}  KGE: {kge(true, pred):.4f}")



        # Fit full models for this horizon
        m10 = qmodel(0.10).fit(X, y)
        m50 = qmodel(0.50).fit(X, y)
        m90 = qmodel(0.90).fit(X, y)

        # save under artifacts/H/
        outdir = ART/str(H)
        outdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(m10, outdir/'model_p10.lgb')
        joblib.dump(m50, outdir/'model_p50.lgb')
        joblib.dump(m90, outdir/'model_p90.lgb')
        meta = {
            "features": features,
            "raster": CFG['raster'],
            "target_col": TARGET,
            "horizon_steps": H
        }
        (outdir/'meta.json').write_text(json.dumps(meta, indent=2))
    (ART/'report.json').write_text(json.dumps(report, indent=2))
    print("Saved per-horizon models and report.json")
