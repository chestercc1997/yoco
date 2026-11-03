#!/usr/bin/env bash
#SBATCH --job-name=yoco-data
#SBATCH --partition=i64m512u
#SBATCH --qos=i64m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the yoco repository}"
source scripts/activate_local.sh
export TOKENIZE_WORKERS="$SLURM_CPUS_PER_TASK"
python - <<'PY'
import json, os, socket, time
from pathlib import Path
p=Path('runs/setup'); p.mkdir(parents=True,exist_ok=True)
(p/'data_job.json').write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],
 'host':socket.gethostname(),'status':'running','started_unix':time.time()},indent=2)+'\n')
PY
trap 'python -c '\''import json; from pathlib import Path; p=Path("runs/setup/data_job.json"); d=json.loads(p.read_text()); d["status"]="failed"; p.write_text(json.dumps(d,indent=2)+"\n")'\''' ERR
bash scripts/prepare_local_data.sh
python - <<'PY'
import json, time
from pathlib import Path
p=Path('runs/setup/data_job.json'); d=json.loads(p.read_text())
d.update(status='verified',completed_unix=time.time())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
