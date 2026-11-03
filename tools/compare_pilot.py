"""Report matched validation points without claiming statistical equivalence."""
import csv
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1])
reference = {int(r['step']): r for r in csv.DictReader((root / 'evaluation/figures/training_efficiency_validation_points.csv').open()) if r['model'] == 'YOCO'}
points = {}
for line in (run / 'rank0.jsonl').read_text().splitlines():
    row = json.loads(line)
    if row['event'] == 'validation' and row['step'] in reference:
        ref = reference[row['step']]
        points[row['step']] = {'step': row['step'], 'tokens': int(ref['tokens']),
                              'loss': row['loss'], 'reference_loss': float(ref['validation_loss']),
                              'loss_delta': row['loss'] - float(ref['validation_loss'])}
result = {'reference': 'repository YOCO FineWeb-Edu public-data reproduction; not original Microsoft run',
          'points': [points[s] for s in sorted(points)],
          'complete_1000_step_pilot': all(s in points for s in (250, 500, 750, 1000)),
          'limitation': 'One seed and changed GPU topology/backend; no final-quality or statistical-equivalence claim.'}
(run / 'baseline_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['complete_1000_step_pilot']:
    raise SystemExit('Missing expected validation points')
