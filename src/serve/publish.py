import pandas as pd, joblib, json, os, yaml
from pathlib import Path

CFG = yaml.safe_load(open('config/config.yaml'))
ART = Path('artifacts')
TARGET = CFG.get('target_col','q_cms')
H_LIST = CFG.get('horizon_steps_list', [24,48,96])

feat = pd.read_parquet('data/processed/feat.parquet').sort_values(['station_id','ts'])
latest = feat.groupby('station_id', group_keys=False).tail(1).copy()

msgs = []
for H in H_LIST:
    outdir = ART/str(H)
    m10 = joblib.load(outdir/'model_p10.lgb')
    m50 = joblib.load(outdir/'model_p50.lgb')
    m90 = joblib.load(outdir/'model_p90.lgb')
    meta = json.loads((outdir/'meta.json').read_text())
    features = meta['features']

    for idx, row in latest.iterrows():
        x = latest.loc[[idx], features]
        msg = {
            "ts": str(row['ts']),
            "station_id": row['station_id'],
            "target": TARGET,
            "horizon_steps": int(H),
            "horizon_minutes": int(H)*15,
            "p10": float(m10.predict(x)[0]),
            "p50": float(m50.predict(x)[0]),
            "p90": float(m90.predict(x)[0])
        }
        msgs.append(msg)

# Save combined list
(ART/'forecast_latest.json').write_text(json.dumps(msgs, indent=2))
print(f"Wrote {len(msgs)} forecast(s) for {len(H_LIST)} horizons -> {ART/'forecast_latest.json'}")

# Optional MQTT publish
if CFG.get('mqtt',{}).get('enabled', False):
    import paho.mqtt.client as mqtt
    client = mqtt.Client()
    client.connect(CFG['mqtt']['host'], int(CFG['mqtt']['port']), 60)
    base_topic = CFG['mqtt']['base_topic'].rstrip('/')
    for msg in msgs:
        topic = f"{base_topic}/{msg['station_id']}/forecast/{msg['horizon_minutes']}min/{TARGET}"
        client.publish(topic, json.dumps(msg), qos=1, retain=False)
    client.disconnect()
    print("Published to MQTT.")
else:
    print("MQTT disabled in config. Only wrote JSON file.")
