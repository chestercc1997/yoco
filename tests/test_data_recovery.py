"""Check preprocessing boundaries, deterministic sampling and checkpoint fallback."""
import hashlib,json,tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from runtime.model import tokenizer
from runtime.train import Tokens,sha256,atomic_json,latest_checkpoint
import tools.prepare_data as prep
with tempfile.TemporaryDirectory() as d:
 root=Path(d);prep.ROOT=root
 val_id=next(str(i) for i in range(10000) if int.from_bytes(hashlib.sha256(str(i).encode()).digest()[:8],'big')%1000==0)
 train_id=next(str(i) for i in range(10000) if int.from_bytes(hashlib.sha256(str(i).encode()).digest()[:8],'big')%1000!=0)
 raw=root/'fixture.parquet';pq.write_table(pa.Table.from_pylist([{'id':val_id,'text':'validation document'},{'id':train_id,'text':'training document'}]),raw)
 row=prep.tokenize_shard((str(raw),'fixture',sha256(raw),'test-revision'))
 tok=tokenizer()
 for s,txt in [('train','training document'),('validation','validation document')]:
  assert np.fromfile(row[s]['path'],dtype='<u4').tolist()==tok.encode(txt,eos=True)
 assert prep.tokenize_shard((str(raw),'fixture',sha256(raw),'test-revision'))==row
 arr=np.arange(4097,dtype='<u4');f=root/'tokens.bin';arr.tofile(f)
 manifest={'shards':[{'train':{'path':str(f),'tokens':len(arr)}}]}
 source=Tokens(manifest,'train',8,42);before=[source.sample(i).tolist() for i in range(512)]
 assert len({tuple(x) for x in before})==512
 restored=Tokens(manifest,'train',8,42)
 assert before[193:]==[restored.sample(i).tolist() for i in range(193,512)]
 for step,content in [(1,b'valid'),(2,b'will corrupt')]:
  p=root/f'checkpoint_{step:08d}.pt';p.write_bytes(content)
  atomic_json(p.with_suffix('.json'),{'fingerprint':'test','file':p.name,'sha256':sha256(p)})
 (root/'checkpoint_00000002.pt').write_bytes(b'corrupted')
 assert latest_checkpoint(root,'test').name=='checkpoint_00000001.pt'
 (root/'checkpoint_00000003.pt.tmp').write_bytes(b'incomplete')
 assert latest_checkpoint(root,'test').name=='checkpoint_00000001.pt'
 try:latest_checkpoint(root,'wrong')
 except RuntimeError:pass
 else:raise AssertionError('Mismatched configuration accepted')
print('PASS preprocessing, disjoint deterministic block shuffle, resume positions, corrupt/partial checkpoint fallback and configuration rejection')
