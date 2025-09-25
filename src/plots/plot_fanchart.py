# src/plots/plot_fanchart.py
"""
Erstellt Fan-Charts (p10/p50/p90) für eine Station & einen Horizont aus Hindcast-Dateien.

Beispiel:
  python -m src.plots.plot_fanchart --station ERFT_001 --horizon 48 \
      --hindcast data/processed/hindcast --out artifacts/plots

Hinweise (Rendering-Regeln):
- Matplotlib verwenden
- Ein Chart pro Figure (keine Subplots)
- Keine Farben explizit setzen
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_fan(df: pd.DataFrame, outpath: Path):
    df = df.sort_values('ts')
    x = pd.to_datetime(df['ts'])

    fig, ax = plt.subplots()
    ax.fill_between(x, df['p10'], df['p90'], alpha=0.3, label='p10–p90')
    ax.plot(x, df['p50'], label='p50')
    if 'y_true' in df.columns:
        ax.plot(x, df['y_true'], label='true', linewidth=1)

    sid = df['station_id'].iloc[0]
    hz = int(df['horizon_minutes'].iloc[0]) if 'horizon_minutes' in df.columns else None
    ax.set_title(f"{sid} – {hz} min" if hz else f"{sid}")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Ziel (z. B. h_cm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True, help="Stations-ID, z. B. ERFT_001")
    ap.add_argument("--horizon", type=int, required=True, help="Horizont in 15-min Schritten")
    ap.add_argument("--hindcast", type=str, default="data/processed/hindcast", help="Verzeichnis mit Hindcast-Parquets")
    ap.add_argument("--out", type=str, default="artifacts/plots", help="Ausgabeverzeichnis für PNG")
    args = ap.parse_args()

    hindir = Path(args.hindcast)
    infile = hindir / f"hindcast_{args.station}_{args.horizon}.parquet"
    if not infile.exists():
        raise SystemExit(f"Nicht gefunden: {infile} – bitte zuerst Hindcast erzeugen.")

    df = pd.read_parquet(infile)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"fanchart_{args.station}_{args.horizon}.png"
    plot_fan(df, outpath)
    print(f"[OK] Fan-Chart -> {outpath}")
