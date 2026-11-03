"""Result publication must reject incomplete or smoke-only evaluations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import watch_plan_ab as watcher


class ReportTests(unittest.TestCase):
    def test_complete_report_and_smoke_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dest=root/'collected';dest.mkdir()
            record=dict(job_id='fixture',release=watcher.BASE+'/experiments/test',runs={arm:watcher.BASE+'/runs/'+arm for arm in ('A','B')})
            evaldir=dest/'experiments/test/runs/eval-plan-ab';evaldir.mkdir(parents=True)
            rows=[]
            for name in ('arc_easy','arc_challenge','boolq','hellaswag','openbookqa','piqa','winogrande','sciq'):
                rows.append(dict(task=name,metric='acc,none',A=.5,B=.6,delta=.1))
                for arm,value in [('A',.5),('B',.6)]:
                    out=evaldir/'results'/arm/name;out.mkdir(parents=True)
                    (out/'results.json').write_text(json.dumps(dict(model_step=40000,smoke_only=False,results={name:{'acc,none':value}})))
                    (out/'provenance.json').write_text(json.dumps(dict(limit=None,dataset_lock={'task':name},task_yaml_sha256=name)))
            (evaldir/'summary.json').write_text(json.dumps(dict(rows=rows,A_macro=.5,B_macro=.6)))
            for arm in ('A','B'):
                out=dest/'runs'/arm;out.mkdir(parents=True)
                train=[dict(event='train',step=i,tokens=i*262144,seconds=1.,peak_memory_bytes=1000) for i in range(1,40001)]
                val=[dict(event='validation',step=i,loss=3.) for i in range(250,40001,250)]
                (out/'rank0.jsonl').write_text('\n'.join(json.dumps(x) for x in train+val))
            (dest/'collection.json').write_text(json.dumps({'sha256':'test'}))
            (root/'README.md').write_text('**Status on 2026-11-03:** pending\n\n- [A/B design](docs/PLAN_AB_4GPU.md)\n')
            (root/'docs').mkdir();(root/'docs/README.md').write_text('pending');(root/'docs/PLAN_AB_4GPU.md').write_text('design')
            with patch.object(watcher,'ROOT',root):
                watcher.report(record,dest)
                result=json.loads((root/'docs/evidence/plan-ab-results.json').read_text())
                self.assertTrue(result['complete']);self.assertEqual(result['training']['A']['final_step'],40000)
                self.assertEqual(len((root/'docs/evidence/plan-ab-validation.csv').read_text().splitlines()),161)
                out=evaldir/'results/A/piqa/results.json';data=json.loads(out.read_text());data['smoke_only']=True;out.write_text(json.dumps(data))
                with self.assertRaises(AssertionError):watcher.report(record,dest)


if __name__=='__main__':unittest.main()
