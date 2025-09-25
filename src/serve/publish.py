# src/serve/publish.py (mode-aware)
import pandas as pd, joblib, json, yaml
from pathlib import Path

CFG = yaml.safe_load(open('config/config.yaml'))
ART = Path('artifacts')
TARGET = CFG.get('target_col','q_cms')
H_LIST = CFG.get('horizon_steps_list',[24,48,96])
MODE = CFG.get('training_mode','global')

feat = pd.read_parquet('data/processed/feat.parquet').sort_values(['station_id','ts'])
latest = feat.groupby('station_id',group_keys=False).tail(1).copy()

msgs=[]

def load_meta_features(path):
    meta = json.loads((path/'meta.json').read_text())
    return meta['features']

if MODE=='global':
    for H in H_LIST:
        outdir=ART/str(H)
        features=load_meta_features(outdir)
        m10=joblib.load(outdir/'model_p10.lgb')
        m50=joblib.load(outdir/'model_p50.lgb')
        m90=joblib.load(outdir/'model_p90.lgb')
        for idx,row in latest.iterrows():
            x=latest.loc[[idx],features]
            msg={
              "ts":str(row['ts']),"station_id":row['station_id'],
              "target":TARGET,"horizon_steps":int(H),"horizon_minutes":int(H)*15,
              "p10":float(m10.predict(x)[0]),"p50":float(m50.predict(x)[0]),"p90":float(m90.predict(x)[0])
            }
            msgs.append(msg)
elif MODE=='per_station':
    for sid,group in latest.groupby('station_id'):
        for H in H_LIST:
            outdir=ART/sid/str(H)
            if not outdir.exists(): continue
            features=load_meta_features(outdir)
            m10=joblib.load(outdir/'model_p10.lgb')
            m50=joblib.load(outdir/'model_p50.lgb')
            m90=joblib.load(outdir/'model_p90.lgb')
            x=group[features].iloc[[-1]]
            msg={
              "ts":str(group['ts'].iloc[-1]),"station_id":sid,
              "target":TARGET,"horizon_steps":int(H),"horizon_minutes":int(H)*15,
              "p10":float(m10.predict(x)[0]),"p50":float(m50.predict(x)[0]),"p90":float(m90.predict(x)[0])
            }
            msgs.append(msg)
else:
    raise SystemExit(f"Unknown training_mode {MODE}")

(ART/'forecast_latest.json').write_text(json.dumps(msgs,indent=2))
print(f"Wrote {len(msgs)} forecasts -> {ART/'forecast_latest.json'}")

if CFG.get('mqtt',{}).get('enabled',False):
    import paho.mqtt.client as mqtt
    client=mqtt.Client()
    client.connect(CFG['mqtt']['host'],int(CFG['mqtt']['port']),60)
    base_topic=CFG['mqtt']['base_topic'].rstrip('/')
    for msg in msgs:
        topic=f"{base_topic}/{msg['station_id']}/forecast/{msg['horizon_minutes']}min/{TARGET}"
        client.publish(topic,json.dumps(msg),qos=1,retain=False)
    client.disconnect()
    print("Published to MQTT.")
else:
    print("MQTT disabled in config. Only wrote JSON file.")
