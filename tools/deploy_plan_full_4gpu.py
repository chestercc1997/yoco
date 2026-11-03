"""Deploy a separately named immutable four-GPU full experiment."""
import fcntl, hashlib, json, shlex, tarfile
from deploy_fusion_2gpu import ROOT, BASE, call, remote

tag='plan-conditional-kv-full-4gpu-v1'; release=BASE+'/experiments/'+tag; setup=ROOT/'runs/setup'
with (setup/(tag+'.lock')).open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files=list((ROOT/'runtime').rglob('*.py'))+list((ROOT/'evaluation').glob('*.py'))
    names=['configs/yoco_160m_plan_full_baseline_4gpu.json','configs/yoco_160m_plan_full_conditional_4gpu.json',
           'tools/preflight_plan_full.py','tools/summarize_full_v2_downstream.py','tests/test_conditional_kv.py',
           'scripts/plan_full_4gpu.sbatch','scripts/plan_full_4gpu_inside.sh']
    files += [ROOT/n for n in names]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    m=setup/(tag+'-manifest.json');m.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=setup/(tag+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
        tar.add(m,arcname='release_manifest.json')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    remote('test ! -e '+shlex.quote(release))
    call(['scp','-o','StrictHostKeyChecking=yes','-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',archive,'jzhou945@hpc3login.hpc.hkust-gz.edu.cn:'+BASE+'/runs/setup/'+archive.name])
    remote('cd '+shlex.quote(BASE)+' && echo '+shlex.quote(sha+'  runs/setup/'+archive.name)+' | sha256sum -c - && mkdir '+shlex.quote(release)+' && tar xzf '+shlex.quote(BASE+'/runs/setup/'+archive.name)+' -C '+shlex.quote(release)+' && ln -s '+shlex.quote(BASE+'/data')+' '+shlex.quote(release+'/data')+' && python3 '+shlex.quote(release+'/tools/preflight_plan_full.py'))
    print('Full release deployed; submission requires a separate check.',flush=True)
