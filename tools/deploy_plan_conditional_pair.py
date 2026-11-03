"""Deploy one immutable 4-GPU paired pilot, with recorded single submission."""
import fcntl
import hashlib
import json
import shlex
import tarfile
from pathlib import Path
from deploy_fusion_2gpu import ROOT, BASE, call, remote

tag = 'plan-conditional-kv-160m-v1'
release = BASE + '/experiments/' + tag
setup = ROOT / 'runs/setup'
with (setup / (tag + '.lock')).open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = list((ROOT/'runtime').rglob('*.py'))
    names = ['configs/yoco_160m_baseline_2gpu.json', 'configs/yoco_160m_conditional_kv_2gpu.json',
             'tools/preflight_fusion.py', 'tests/test_conditional_kv.py',
             'scripts/plan_conditional_pair_inside.sh', 'scripts/plan_conditional_pair.sbatch']
    files += [ROOT/n for n in names]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    m = setup/(tag+'-manifest.json'); m.write_text(json.dumps(manifest, indent=2)+'\n')
    archive=setup/(tag+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files: tar.add(p, arcname=str(p.relative_to(ROOT)))
        tar.add(m, arcname='release_manifest.json')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    remote('test ! -e '+shlex.quote(release))
    call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',archive,'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive.name])
    remote('cd '+shlex.quote(BASE)+' && echo '+shlex.quote(sha+'  runs/setup/'+archive.name)+' | sha256sum -c - && mkdir '+shlex.quote(release)+' && tar xzf '+shlex.quote(BASE+'/runs/setup/'+archive.name)+' -C '+shlex.quote(release)+' && ln -s '+shlex.quote(BASE+'/data')+' '+shlex.quote(release+'/data')+' && python3 '+shlex.quote(release+'/tools/preflight_fusion.py')+' --config configs/yoco_160m_conditional_kv_2gpu.json')
    print('Deployed and verified; submission is a separate audited step.', flush=True)
