# src/models/train_baseline.py (mode-aware)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.utils.metrics import nse, kge, rmse as rmse_fn

CFG = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
INP = Path('data/processed/feat.parquet')
ART = Path('artifacts'); ART.mkdir(parents=True, exist_ok=True)

TARGET = CFG.get('target_col', 'q_cms')
H_LIST = CFG.get('horizon_steps_list', [24, 48, 96])
MODE = CFG.get('training_mode', 'global')

def qmodel(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective='quantile', alpha=alpha,
        n_estimators=700, learning_rate=0.03,
        num_leaves=96, subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )

def features_from(df: pd.DataFrame):
    base_exclude = {'ts','station_id','q_cms','h_cm','y'}
    return [c for c in df.columns if c not in base_exclude]

if __name__ == "__main__":
    df = pd.read_parquet(INP).sort_values(['station_id','ts'])
    features = features_from(df)
    report = {}

    if MODE == 'global':
        for H in H_LIST:
            d = df.copy()
            d['y'] = d.groupby('station_id')[TARGET].shift(-H)
            d = d.dropna(subset=['y'])
            X = d[features]; y = d['y']
            tscv = TimeSeriesSplit(n_splits=5)
            preds, trues = [], []
            for tr, te in tscv.split(X):
                m_cv = qmodel(0.5).fit(X.iloc[tr], y.iloc[tr])
                p_cv = m_cv.predict(X.iloc[te])
                preds.append(p_cv); trues.append(y.iloc[te].values)
            pred = np.concatenate(preds); true = np.concatenate(trues)
            mae = mean_absolute_error(true, pred)
            rmse_val = rmse_fn(true, pred)
            nse_val = nse(true, pred)
            kge_val = kge(true, pred)
            print(f"[GLOBAL] H={H} MAE:{mae:.4f} RMSE:{rmse_val:.4f} NSE:{nse_val:.4f} KGE:{kge_val:.4f}")
            report[f"global_{H}"] = {"MAE": float(mae),"RMSE": float(rmse_val),"NSE": float(nse_val),"KGE": float(kge_val)}
            outdir = ART/str(H); outdir.mkdir(parents=True, exist_ok=True)
            m10 = qmodel(0.10).fit(X,y); joblib.dump(m10, outdir/'model_p10.lgb')
            m50 = qmodel(0.50).fit(X,y); joblib.dump(m50, outdir/'model_p50.lgb')
            m90 = qmodel(0.90).fit(X,y); joblib.dump(m90, outdir/'model_p90.lgb')
            meta = {"features": features,"raster": CFG['raster'],"target_col": TARGET,"horizon_steps": H,"mode":"global"}
            (outdir/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    elif MODE == 'per_station':
        stations = df['station_id'].dropna().unique().tolist()
        for H in H_LIST:
            for sid in stations:
                d = df[df['station_id']==sid].copy()
                d['y'] = d[TARGET].shift(-H)
                d = d.dropna(subset=['y'])
                if len(d)<120: continue
                X = d[features]; y = d['y']
                tscv = TimeSeriesSplit(n_splits=3 if len(d)<500 else 5)
                preds, trues = [], []
                for tr, te in tscv.split(X):
                    m_cv = qmodel(0.5).fit(X.iloc[tr], y.iloc[tr])
                    p_cv = m_cv.predict(X.iloc[te])
                    preds.append(p_cv); trues.append(y.iloc[te].values)
                pred = np.concatenate(preds); true = np.concatenate(trues)
                mae = mean_absolute_error(true, pred)
                rmse_val = rmse_fn(true, pred)
                nse_val = nse(true, pred)
                kge_val = kge(true, pred)
                print(f"[PER_STATION] {sid} H={H} MAE:{mae:.4f} RMSE:{rmse_val:.4f} NSE:{nse_val:.4f} KGE:{kge_val:.4f}")
                report[f"{sid}_{H}"] = {"MAE": float(mae),"RMSE": float(rmse_val),"NSE": float(nse_val),"KGE": float(kge_val)}
                outdir = ART/sid/str(H); outdir.mkdir(parents=True, exist_ok=True)
                m10 = qmodel(0.10).fit(X,y); joblib.dump(m10,outdir/'model_p10.lgb')
                m50 = qmodel(0.50).fit(X,y); joblib.dump(m50,outdir/'model_p50.lgb')
                m90 = qmodel(0.90).fit(X,y); joblib.dump(m90,outdir/'model_p90.lgb')
                meta = {"features":features,"raster":CFG['raster'],"target_col":TARGET,"horizon_steps":H,"mode":"per_station","station_id":sid}
                (outdir/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    else:
        raise SystemExit(f"Unknown training_mode {MODE}")

    (ART/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print("Training complete.")
