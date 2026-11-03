"""One model entry point for YOCO and the paired Transformer baseline."""

from .tokenizer import TiktokenTokenizer
from .yoco_official import YOCO, YOCOArgs
from .yoco_official.transformer import ModelArgs, Transformer
from .memory_fusion import FusionConfig, MemoryFusionYOCO


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
        if 'memory_fusion' in config:
            args = YOCOArgs(**config['model'])
            fusion = FusionConfig.parse(config['memory_fusion'], args)
            return MemoryFusionYOCO(args, fusion, **common)
        return YOCO(YOCOArgs(**config["model"]), **common)
    if 'memory_fusion' in config:
        raise ValueError('memory_fusion is only supported for YOCO')
    return Transformer(ModelArgs(**config["model"]), **common)
