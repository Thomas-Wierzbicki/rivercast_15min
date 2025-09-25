# src/plots/batch_render.py
"""
Batch-Rendering für Fan-Charts (p10/p50/p90) über alle Stationen × Horizonte.

Voraussetzungen:
- Hindcast-Dateien existieren unter data/processed/hindcast/hindcast_{station}_{H}.parquet
  (Erstellt mit: python -m src.eval.hindcast)

Nutzung:
  python -m src.plots.batch_render --hindcast data/processed/hindcast --out artifacts/plots
  # Optional: nur bestimmte Stationen/Horizonte
  python -m src.plots.batch_render --stations ERFT_001 ERFT_002 --horizons 24 48

Erzeugt PNGs: fanchart_{station}_{H}.png
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
    hz = int(df['horizon_minutes'].iloc[0]) if 'horizon_minutes' in df.columns else int(df['horizon_steps'].iloc[0])*15
    ax.set_title(f"{sid} – {hz} min")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Ziel (z. B. h_cm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", type=str, default="data/processed/hindcast", help="Verzeichnis mit Hindcast-Parquets")
    ap.add_argument("--out", type=str, default="artifacts/plots", help="Ausgabeverzeichnis für PNGs")
    ap.add_argument("--stations", nargs="*", default=None, help="Filter: Stations-IDs")
    ap.add_argument("--horizons", nargs="*", type=int, default=None, help="Filter: Horizonte (15-min Schritte)")
    args = ap.parse_args()

    hindir = Path(args.hindcast)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    files = list(hindir.glob("hindcast_*.parquet"))
    if not files:
        raise SystemExit(f"Keine Hindcast-Dateien in {hindir} gefunden. Bitte zuerst: python -m src.eval.hindcast")

    for f in files:
        # Dateiname: hindcast_{station}_{H}.parquet
        try:
            stem = f.stem  # e.g., hindcast_ERFT_001_48
            parts = stem.split("_")
            station = "_".join(parts[1:-1])
            horizon = int(parts[-1])
        except Exception:
            continue

        if args.stations and station not in args.stations:
            continue
        if args.horizons and horizon not in args.horizons:
            continue

        df = pd.read_parquet(f)
        png = outdir / f"fanchart_{station}_{horizon}.png"
        plot_fan(df, png)
        print(f"[OK] {png}")
