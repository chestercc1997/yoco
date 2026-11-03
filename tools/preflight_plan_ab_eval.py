"""Load all locked downstream datasets offline before queueing the campaign."""
import json
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
evalroot=root.parents[1]/'eval-20261102'
sys.path.insert(0,str(evalroot/'site'))
sys.path.insert(0,str(root))
os.environ.update(HF_HOME=str(evalroot/'data/cache/huggingface'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_DATASETS_TRUST_REMOTE_CODE='1',TOKENIZERS_PARALLELISM='false')
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config
from evaluation.locked_task import apply_lock
manager=TaskManager()
for name in ('arc_easy','arc_challenge','boolq','hellaswag','openbookqa','piqa','winogrande','sciq'):
    config=load_yaml_config(manager.task_index[name]['yaml_path'])
    lock=json.loads((root/f'evaluation/plan_ab_data_locks/data_lock_{name}.json').read_text())
    old=json.loads((evalroot/f'runs/eval-full-v2/data_locks/data_lock_{name}.json').read_text())
    assert lock==old,name
    apply_lock(config,lock,os.environ['HF_HOME'])
    task=ConfigurableTask(config=config)
    docs=task.test_docs() if task.has_test_docs() else task.validation_docs()
    assert len(docs)>0,name
    print(f'PASS offline {name}: {len(docs)} examples',flush=True)
