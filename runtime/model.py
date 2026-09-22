"""One model entry point for YOCO and the paired Transformer baseline."""

from .tokenizer import TiktokenTokenizer
from .yoco_official import YOCO, YOCOArgs
from .yoco_official.transformer import ModelArgs, Transformer


def tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer("cl100k_base")


def architecture(config: dict) -> str:
    value = config.get("architecture")
    if value not in {"yoco", "transformer"}:
        raise ValueError("config.architecture must be 'yoco' or 'transformer'")
    return value


def build_model(config: dict, checkpoint_activations: bool = False):
    common = {
        "checkpoint_activations": checkpoint_activations,
        "share_input_output_embed": config["tied_embeddings"],
    }
    if architecture(config) == "yoco":
        return YOCO(YOCOArgs(**config["model"]), **common)
    return Transformer(ModelArgs(**config["model"]), **common)
