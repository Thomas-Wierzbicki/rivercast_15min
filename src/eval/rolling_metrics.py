# src/eval/rolling_metrics.py
"""
Berechnet rollende Skill-Metriken (MAE, NSE, KGE) und Coverage (y_true in [p10,p90]) aus Hindcast-Dateien.

Nutzung:
  python -m src.eval.rolling_metrics --hindcast data/processed/hindcast --window 96 --out data/processed/metrics

- window = Anzahl Punkte im Rollfenster (z. B. 96 bei 24h auf 15-min Raster)
Ergebnisse:
  - Parquet/CSV pro Station/Horizont mit Spalten: ts, station_id, horizon_steps, mae, nse, kge, coverage
  - Optional: einfache Plot-Ausgabe (je Metrik) via --plot-out artifacts/metrics_plots
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mae(y, yhat): return np.mean(np.abs(y - yhat)) if len(y)>0 else np.nan
def nse(y, yhat):
    if len(y)==0: return np.nan
    denom = np.sum((y - np.mean(y))**2)
    if denom == 0: return np.nan
    return 1 - np.sum((y - yhat)**2) / denom
def kge(y, yhat):
    if len(y)==0: return np.nan
    r = np.corrcoef(y, yhat)[0,1] if len(y)>1 else np.nan
    alpha = np.std(yhat)/np.std(y) if np.std(y)>0 else np.nan
    beta = np.mean(yhat)/np.mean(y) if np.mean(y)!=0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def coverage(y, p10, p90):
    return np.mean((y >= p10) & (y <= p90)) if len(y)>0 else np.nan

def plot_metric(df, col, outpath: Path, title: str):
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(df['ts']), df[col])
    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", type=str, default="data/processed/hindcast")
    ap.add_argument("--window", type=int, default=96, help="Fenstergröße (Anzahl Punkte)")
    ap.add_argument("--out", type=str, default="data/processed/metrics")
    ap.add_argument("--plot-out", type=str, default=None, help="Optionales Plot-Verzeichnis")
    args = ap.parse_args()

    hindir = Path(args.hindcast)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    plotdir = Path(args.plot_out) if args.plot_out else None
    if plotdir: plotdir.mkdir(parents=True, exist_ok=True)

    files = list(hindir.glob("hindcast_*.parquet"))
    if not files:
        raise SystemExit(f"Keine Hindcast-Dateien in {hindir} gefunden.")

    for f in files:
        df = pd.read_parquet(f).sort_values('ts').reset_index(drop=True)
        sid = df['station_id'].iloc[0]
        H = int(df['horizon_steps'].iloc[0]) if 'horizon_steps' in df.columns else int(df['horizon_minutes'].iloc[0])//15

        y = df['y_true'].to_numpy()
        yhat = df['p50'].to_numpy()
        p10 = df['p10'].to_numpy()
        p90 = df['p90'].to_numpy()

        # Rolling arrays
        n = len(y)
        w = args.window
        roll_mae = [np.nan]*n
        roll_nse = [np.nan]*n
        roll_kge = [np.nan]*n
        roll_cov = [np.nan]*n

        for i in range(w, n+1):
            sl = slice(i-w, i)
            roll_mae[i-1] = mae(y[sl], yhat[sl])
            roll_nse[i-1] = nse(y[sl], yhat[sl])
            roll_kge[i-1] = kge(y[sl], yhat[sl])
            roll_cov[i-1] = coverage(y[sl], p10[sl], p90[sl])

        out = pd.DataFrame({
            "ts": df["ts"],
            "station_id": sid,
            "horizon_steps": H,
            "window": w,
            "mae": roll_mae,
            "nse": roll_nse,
            "kge": roll_kge,
            "coverage": roll_cov
        })

        out_parquet = outdir / f"metrics_{sid}_{H}_w{w}.parquet"
        out_csv = outdir / f"metrics_{sid}_{H}_w{w}.csv"
        out.to_parquet(out_parquet)
        out.to_csv(out_csv, index=False)
        print(f"[OK] Saved metrics -> {out_parquet.name}, {out_csv.name}")

        if plotdir:
            plot_metric(out, "mae", plotdir / f"mae_{sid}_{H}_w{w}.png", f"MAE – {sid} – H={H}")
            plot_metric(out, "nse", plotdir / f"nse_{sid}_{H}_w{w}.png", f"NSE – {sid} – H={H}")
            plot_metric(out, "kge", plotdir / f"kge_{sid}_{H}_w{w}.png", f"KGE – {sid} – H={H}")
            plot_metric(out, "coverage", plotdir / f"coverage_{sid}_{H}_w{w}.png", f"Coverage(p10–p90) – {sid} – H={H}")
