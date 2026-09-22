"""Prepare a pinned public corpus; never runs without explicit dataset approval."""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.train import atomic_json, sha256


def tokenize_shard(task):
    import numpy as np
    import pyarrow.parquet as pq
    from runtime.model import tokenizer
    raw, name, expected, revision=task
    directory=ROOT/'data/fineweb_edu_10bt/tokens'; directory.mkdir(parents=True,exist_ok=True)
    done=directory/(name+'.json')
    if done.exists():
        record=json.loads(done.read_text())
        if record['source_sha256']==expected and record['revision']==revision and all(Path(record[s]['path']).is_file() and sha256(record[s]['path'])==record[s]['sha256'] for s in ('train','validation')):
            return record
    tok=tokenizer(); paths={s:directory/(name+'.'+s+'.bin') for s in ('train','validation')}
    handles={s:open(str(p)+'.tmp','wb') for s,p in paths.items()}; counts={s:0 for s in paths}; docs={s:0 for s in paths}; empty=0
    try:
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=256,columns=['id','text']):
            for row in batch.to_pylist():
                if not row['text']: empty+=1; continue
                split='validation' if int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8],'big')%1000==0 else 'train'
                ids=tok.encode(row['text'],eos=True)
                arr=np.asarray(ids,dtype='<u4'); handles[split].write(arr.tobytes()); counts[split]+=len(ids); docs[split]+=1
        for s,f in handles.items(): f.flush(); os.fsync(f.fileno()); f.close(); os.replace(str(paths[s])+'.tmp',paths[s])
    finally:
        for f in handles.values(): f.close()
    record={'source_file':str(raw),'source_sha256':expected,'revision':revision,'skipped_empty_documents':empty}
    for s,p in paths.items(): record[s]={'path':str(p),'tokens':counts[s],'documents':docs[s],'sha256':sha256(p)}
    atomic_json(done,record); print(json.dumps({'event':'tokenized','shard':name,'counts':counts}),flush=True)
    return record


def main():
    from huggingface_hub import hf_hub_download
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    c=json.loads(args.config.read_text())
    if not c['data']['approved']: raise SystemExit('Data selection is pending; no training corpus downloaded.')
    candidate=json.loads((ROOT/'provenance/fineweb_candidate.json').read_text())
    if c['data']['dataset']!=candidate['id'] or c['data']['subset']!='sample-10BT': raise SystemExit('Candidate mismatch')
    revision=candidate['sha']; repo=candidate['id']; base=ROOT/'data/fineweb_edu_10bt';base.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/sample/10BT?limit=100'
    tree=json.load(urllib.request.urlopen(url,timeout=60)); entries={x['path']:x for x in tree if x['type']=='file'}
    inventory=[]
    for path in candidate['files']:
        x=entries[path];inventory.append({'path':path,'bytes':x['size'],'sha256':x['lfs']['oid']})
    if inventory!=candidate['inventory']:raise RuntimeError('Pinned source inventory changed')
    import shutil
    # Reserve room for downloaded parquet, uint32 tokens and safe working space.
    required=sum(x['bytes'] for x in inventory)+(55<<30)
    if not (base/'manifest.json').exists() and shutil.disk_usage(base).free<required:
        raise RuntimeError(f'Insufficient disk space for data preparation: need {required} bytes free')
    atomic_json(base/'raw_manifest.json',{'dataset':repo,'revision':revision,'files':inventory})
    tasks=[]
    for entry in inventory:
        for attempt in range(8):
            try:
                raw=hf_hub_download(repo_id=repo,repo_type='dataset',revision=revision,filename=entry['path'],local_dir=str(base/'raw'))
                if Path(raw).stat().st_size!=entry['bytes'] or sha256(raw)!=entry['sha256']: raise RuntimeError('Raw data checksum mismatch')
                break
            except Exception:
                if attempt==7: raise
                time.sleep(min(60,5*2**attempt))
        tasks.append((raw,Path(entry['path']).stem,entry['sha256'],revision)); print(json.dumps({'event':'downloaded','file':entry['path']}),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool: records=list(pool.map(tokenize_shard,tasks))
    manifest={'ready':True,'dataset':repo,'subset':'sample-10BT','revision':revision,'shards':records,
              'dtype':'little_endian_uint32','tokenizer':'official YOCO cl100k_base wrapper, vocab=100288, normal IDs+4, EOS=2',
              'preprocessing':'Encode complete original text with official wrapper and append EOS; pack within each source shard. No new text filtering except empty documents.',
              'split_rule':'sha256(id)[:8] as unsigned big endian modulo 1000 == 0 for validation; all other documents train',
              'tokenization_script_sha256':sha256(__file__),
              'original_yoco_data':False,'totals':{s:sum(x[s]['tokens'] for x in records) for s in ('train','validation')}}
    atomic_json(base/'manifest.json',manifest)
    c['data']['manifest']=str(base/'manifest.json');c['data']['revision']=revision;c['status']='DATA_READY_RUNTIME_VALIDATION_REQUIRED'
    atomic_json(args.config,c)
    print(json.dumps({'event':'data_ready','totals':manifest['totals']}),flush=True)

if __name__=='__main__':main()
