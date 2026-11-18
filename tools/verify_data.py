import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.train import sha256,atomic_json
import argparse

parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);a=parser.parse_args()
c=json.loads(a.config.read_text())
manifest=Path(c['data']['manifest'])
if not manifest.is_absolute():
    base = ROOT if c['data'].get('packing') == 'yoco_samples' else a.config.resolve().parents[1]
    manifest = base / manifest
m=json.loads(manifest.read_text());assert m['ready']
if m.get('packing','stream') != c['data'].get('packing','stream'):
    raise RuntimeError('Data packing differs from config')
if m.get('packing') == 'yoco_samples' and m.get('sequence_length') != c['training']['sequence_length']:
    raise RuntimeError('Packed record length differs from config')
for row in m['shards']:
    for split in ('train','validation'):
        x=row[split];p=Path(x['path'])
        if p.stat().st_size!=x['tokens']*4 or sha256(p)!=x['sha256']:raise RuntimeError('Dataset hash mismatch: '+str(p))
print(json.dumps({'verified':True,'dataset':m['dataset'],'revision':m['revision'],'totals':m['totals']}))
