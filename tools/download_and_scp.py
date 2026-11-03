"""Local-only resumable download, then SCP and remote SHA256 verification.

No Slurm submission, tokenization or training. Credentials remain in the existing
HPC3 skill; SSH uses its authorized askpass helper and known-host verification.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT.parent / '.codex-home/skills/hpc3'
REMOTE = '/data/share/jzhou945/cchen099/yoco/data/fineweb_edu_10bt'
STATE = ROOT / 'runs/setup/local_download_transfer.json'


def save(state, **updates):
    state.update(updates, updated_unix=time.time())
    tmp = STATE.with_suffix('.new')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def command(client, argv):
    _, out, err = client.exec_command(shlex.join(argv), timeout=900)
    stdout = out.read().decode(); stderr = err.read().decode()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError(f'Remote command exited {code}: {stderr[-1000:]}')
    return stdout


def main():
    ROOT.joinpath('runs/setup').mkdir(parents=True, exist_ok=True)
    with (ROOT/'runs/setup/.local-download-transfer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='DOWNLOADING_LOCALLY', pid=os.getpid(), started_unix=time.time(),
                     cancelled_slurm_job='13393990', slurm_submission=False,
                     destination=REMOTE, files={}, tokenization_started=False)
        save(state)
        client = None
        try:
            env = dict(os.environ)
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env.pop(key, None)
            env.update(NO_PROXY='*', no_proxy='*')
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/download_pinned_data.py'),
                            '--endpoint', 'https://hf-mirror.com', '--workers', '2'],
                           cwd=ROOT, env=env, check=True)
            inventory = json.loads((ROOT/'provenance/fineweb_candidate.json').read_text())
            save(state, status='LOCAL_DOWNLOAD_SHA256_VERIFIED', file_count=len(inventory['inventory']),
                 total_bytes=inventory['total_raw_bytes'], revision=inventory['sha'])
            auth = json.loads((SKILL/'private/auth.json').read_text())
            host, username = auth['host'], auth['username']
            client = paramiko.SSHClient(); client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                client.connect(host, username=username, password=auth['password'],
                               look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20)
            finally:
                auth.clear()
            client.get_transport().set_keepalive(30)
            env.update(SSH_ASKPASS=str(SKILL/'scripts/askpass.py'), SSH_ASKPASS_REQUIRE='force', DISPLAY='hpc3-auth')
            save(state, status='SCP_IN_PROGRESS')
            for entry in inventory['inventory']:
                rel = entry['path']; local = ROOT/'data/fineweb_edu_10bt/raw'/rel
                remote = REMOTE + '/raw/' + rel
                command(client, ['mkdir','-p',str(Path(remote).parent)])
                with client.open_sftp() as sftp:
                    try: sftp.stat(remote); exists = True
                    except FileNotFoundError: exists = False
                if exists:
                    actual = command(client, ['sha256sum','--',remote]).split()[0]
                    if actual != entry['sha256']:
                        raise RuntimeError('Existing remote file differs; preserve and inspect: ' + remote)
                    method = 'verified_existing'
                else:
                    temporary = remote + '.hpc2-20261029.uploading'
                    save(state, current_file=rel)
                    print(json.dumps(dict(event='scp_start', file=rel)), flush=True)
                    subprocess.run(['scp','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
                        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
                        '-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',
                        str(local), username+'@'+host+':'+temporary],
                        env=env, check=True, timeout=3600, stdin=subprocess.DEVNULL, start_new_session=True)
                    actual = command(client, ['sha256sum','--',temporary]).split()[0]
                    if actual != entry['sha256']:
                        raise RuntimeError('SCP SHA256 mismatch: ' + rel)
                    # Do not overwrite a final file that appeared concurrently.
                    command(client, ['python3','-c',
                        'import os,sys; os.link(sys.argv[1],sys.argv[2]); os.unlink(sys.argv[1])', temporary, remote])
                    method = 'scp_and_remote_sha256_verified'
                state['files'][rel] = dict(sha256=actual, bytes=entry['bytes'], method=method)
                save(state)
                print(json.dumps(dict(event='remote_verified', file=rel, method=method)), flush=True)
            provenance = dict(dataset=inventory['id'], revision=inventory['sha'], files=inventory['inventory'],
                              raw_data_verified=True, tokenized_data_ready=False)
            manifest = ROOT/'runs/setup/transferred_raw_manifest.json'
            manifest.write_text(json.dumps(provenance, indent=2)+'\n')
            subprocess.run(['scp','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
                '-o','GSSAPIAuthentication=no','-o','NumberOfPasswordPrompts=1',str(manifest),
                username+'@'+host+':'+REMOTE+'/hpc2_raw_transfer_manifest.json'],
                env=env, check=True, timeout=120, stdin=subprocess.DEVNULL, start_new_session=True)
            assert command(client,['sha256sum','--',REMOTE+'/hpc2_raw_transfer_manifest.json']).split()[0] == hashlib.sha256(manifest.read_bytes()).hexdigest()
            save(state, status='SCP_AND_SHA256_VERIFIED', completed_unix=time.time(), current_file=None)
        except Exception as exc:
            save(state, status='FAILED_REQUIRES_INSPECTION', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            if client: client.close()


if __name__ == '__main__': main()
