"""Apply dataset locks, including an offline parquet snapshot for PIQA."""
from pathlib import Path


def apply_lock(config, lock, hf_home):
    if lock["task"] == "piqa" and lock["revision"] == "b40a463b4ac9c77c6e74e23c5d873f5c2139d8fa":
        snapshot = Path(hf_home) / "hub/datasets--ybisk--piqa/snapshots" / lock["revision"] / "plain_text"
        config["dataset_path"] = "parquet"
        config["dataset_name"] = None
        config["dataset_kwargs"] = {"data_files": {split: str(snapshot / f"{split}-00000-of-00001.parquet") for split in ("train", "validation", "test")}}
    else:
        config["dataset_path"] = lock["dataset"]
        config["dataset_kwargs"] = {**config.get("dataset_kwargs", {}), "revision": lock["revision"], "trust_remote_code": True}
