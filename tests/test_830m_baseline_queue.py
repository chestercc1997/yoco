"""A queued baseline cannot start on training-only or partial downstream completion."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_830m_baseline_queue as queue


class QueueTests(unittest.TestCase):
    def test_full_predecessor_pipeline_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root / 'baseline').mkdir();(root / 'report').mkdir()
            b = dict(complete=True, step=40000, world_size=8, architecture='YOCO-SWA', parameters=838315008,
                     tasks=[dict(task=k,documents=v) for k,v in queue.TASK_COUNTS.items()])
            queue.atomic_json(root / 'baseline/summary.json', b);(root / 'baseline/PIPELINE_SUCCESS').touch()
            self.assertFalse(queue.predecessor_ready(root))
            queue.atomic_json(root / 'summary.json', dict(complete=True,report_output=str(root / 'report')))
            (root / 'PIPELINE_SUCCESS').touch()
            self.assertFalse(queue.predecessor_ready(root))
            queue.atomic_json(root / 'report/summary.json', dict(complete=True))
            self.assertTrue(queue.predecessor_ready(root))
            b['tasks'][0]['documents'] = 4
            queue.atomic_json(root / 'baseline/summary.json', b)
            self.assertFalse(queue.predecessor_ready(root))

    def test_busy_or_missing_gpus_cannot_release_queue(self):
        for usage,expected in (('0\n'*8,True),('0\n'*7,False),('0\n'*7+'36830\n',False)):
            with patch.object(queue.subprocess,'check_output',return_value=usage):
                self.assertEqual(queue.gpus_idle(),expected)

    def test_existing_guard_is_found_without_spawning_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',
                str(queue.ROOT / 'tools/watch_gret_830m_baseline.py'),'--run-dir',str(run)])
            try:
                found = queue.existing_guard(run)
                self.assertEqual(found['pid'],child.pid)
                self.assertIsNone(queue.existing_guard(run / 'other'))
            finally:
                child.terminate();child.wait(timeout=5)


if __name__ == '__main__':unittest.main()
