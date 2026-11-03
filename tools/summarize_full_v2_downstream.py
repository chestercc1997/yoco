"""Summarize paired full-task results, rejecting smoke or mismatched runs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "runs/eval-full-v2/results"
metrics = {"arc_easy": "acc_norm,none", "arc_challenge": "acc_norm,none", "boolq": "acc,none", "hellaswag": "acc_norm,none", "openbookqa": "acc_norm,none", "piqa": "acc_norm,none", "winogrande": "acc,none", "sciq": "acc_norm,none"}
rows = []
for task, metric in metrics.items():
    data = []
    for arm in ("baseline", "dynamic"):
        folder = root / arm / task
        result = json.loads((folder / "results.json").read_text())
        prov = json.loads((folder / "provenance.json").read_text())
        assert result["model_step"] == 40000 and not result["smoke_only"] and prov["limit"] is None
        data.append((result, prov))
    assert data[0][1]["dataset_lock"] == data[1][1]["dataset_lock"]
    assert data[0][1]["task_yaml_sha256"] == data[1][1]["task_yaml_sha256"]
    base = data[0][0]["results"][task][metric]
    dyn = data[1][0]["results"][task][metric]
    rows.append({"task": task, "metric": metric, "baseline": base, "dynamic": dyn, "delta": dyn - base})
out = root.parent / "summary.json"
out.write_text(json.dumps({"rows": rows, "baseline_macro": sum(x["baseline"] for x in rows) / len(rows), "dynamic_macro": sum(x["dynamic"] for x in rows) / len(rows)}, indent=2) + "\n")
print(out)
