"""Create deterministic, untrained baseline/A/B initial weights and tokenizer cache."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/initial")
    args = parser.parse_args()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "data/cache/tiktoken"))
    import numpy as np
    import torch
    from runtime.model import build_model, tokenizer
    from runtime.train import atomic_json, sha256

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    assert tok.n_words == 100288 and tok.eos_id == 2
    text = "YOCO baseline reproducibility. 你好，世界！"
    assert tok.decode(tok.encode(text, eos=True)) == text
    report = {"pretrained": False, "seed": 42, "purpose": "from-scratch training initialization",
              "tokenizer": {"name": "cl100k_base", "vocab_size": tok.n_words, "eos_id": tok.eos_id,
                            "normal_id_offset": 4, "source_sha256": sha256(ROOT / "runtime/tokenizer.py")},
              "models": {}}
    common_hashes = None
    ab_logits = None
    for name, filename in (("baseline", "yoco_160m_plan_full_baseline_4gpu.json"),
                           ("A", "yoco_160m_plan_ab_A_4gpu.json"),
                           ("B", "yoco_160m_plan_ab_B_4gpu.json")):
        config_path = ROOT / "configs" / filename
        config = json.loads(config_path.read_text())
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = build_model(config).eval()
        state = model.state_dict()
        hashes = {key: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
                  for key, value in state.items() if not key.startswith("memory_fusion.")}
        if common_hashes is None:
            common_hashes = hashes
        else:
            assert hashes == common_hashes, f"Backbone initialization differs: {name}"
        with torch.no_grad():
            logits = model(torch.tensor([tok.encode("Reproducible YOCO model.")]))[0]
            assert torch.isfinite(logits).all()
            if name == "A":
                ab_logits = logits.clone()
            elif name == "B":
                torch.testing.assert_close(logits, ab_logits, rtol=0, atol=0)
        path = args.output / f"{name}-step00000000.pt"
        temporary = path.with_suffix(".pt.tmp")
        with temporary.open("wb") as handle:
            torch.save({"model": state, "model_config": config["model"],
                        "tied_embeddings": config["tied_embeddings"], "step": 0, "seed": config["seed"],
                        "pretrained": False, "memory_fusion": config.get("memory_fusion")}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        report["models"][name] = {"config": str(config_path.relative_to(ROOT)),
                                  "config_sha256": sha256(config_path), "checkpoint": str(path),
                                  "sha256": sha256(path), "parameters": sum(p.numel() for p in model.parameters()),
                                  "forward_finite": True, "common_backbone_verified": True}
        print(json.dumps({"ready_model": name, **report["models"][name]}), flush=True)
        del model, state, logits
        gc.collect()
    report["ab_initial_outputs_identical"] = True
    report["ready"] = True
    atomic_json(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
