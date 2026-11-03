"""Experimental causal depth fusion for YOCO's shared memory, not layer skipping."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .yoco_official import YOCO


@dataclass(frozen=True)
class FusionConfig:
    mode: str
    source_layers: tuple
    temperature: float = 1.0
    initial_weights: tuple = None

    @classmethod
    def parse(cls, value, model_args):
        if not isinstance(value, dict):
            raise ValueError('memory_fusion must be a mapping')
        unknown = set(value) - {'mode', 'source_layers', 'temperature', 'initial_weights'}
        if unknown:
            raise ValueError(f'Unknown memory_fusion fields: {sorted(unknown)}')
        mode = value.get('mode', 'last')
        if mode not in {'last', 'mean', 'static', 'token', 'learned_kv', 'conditional_kv'}:
            raise ValueError('Unknown fusion mode')
        layers = value.get('source_layers', [model_args.n_layers // 2])
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError('source_layers must be a nonempty ordered list')
        depth = model_args.n_layers // 2
        if any(type(i) is not int or not 1 <= i <= depth for i in layers):
            raise ValueError('source_layers are one-based self-decoder layer indices')
        if list(layers) != sorted(set(layers)) or layers[-1] != depth:
            raise ValueError('source_layers must be unique, ascending, and include the final self layer')
        if model_args.n_layers % 2 or model_args.sliding_window is not None:
            raise ValueError('Fusion v1 requires even-depth YOCO-gRet')
        temperature = value.get('temperature', 1.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        initial = value.get('initial_weights')
        if initial is not None:
            if mode not in {'token', 'static'}:
                raise ValueError('initial_weights requires token or static fusion')
            if not isinstance(initial, (list, tuple)) or len(initial) != len(layers):
                raise ValueError('initial_weights must match source_layers')
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0 for p in initial):
                raise ValueError('initial_weights must be finite and strictly positive')
            if not math.isclose(sum(initial), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError('initial_weights must sum to one')
            initial = tuple(float(p) for p in initial)
        return cls(mode, tuple(layers), float(temperature), initial)


class DepthMemoryFusion(nn.Module):
    def __init__(self, dim, eps, config):
        super().__init__()
        self.config, self.eps = config, eps
        count = len(config.source_layers)
        if config.mode == 'conditional_kv':
            self.score = nn.Parameter(torch.zeros(dim))
            self.layer_bias = nn.Parameter(torch.zeros(count))
        elif config.mode == 'token':
            self.router = nn.Linear(dim, count)
            # All mixture controls start uniformly; no routing randomness.
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        elif config.mode in {'static', 'learned_kv'}:
            self.logits = nn.Parameter(torch.zeros(count))
        if config.initial_weights is not None:
            target = self.router.bias if config.mode == 'token' else self.logits
            # softmax(temperature * log(prior) / temperature) = prior.
            with torch.no_grad():
                target.copy_(target.new_tensor([config.temperature * math.log(p) for p in config.initial_weights]))

    def routing_weights(self, final):
        """Return weights without retaining tensors on the module between forwards."""
        count = len(self.config.source_layers)
        if self.config.mode == 'last':
            weights = final.new_zeros(count); weights[-1] = 1
            return weights
        if self.config.mode == 'mean':
            return final.new_full((count,), 1 / count)
        if self.config.mode == 'static':
            return (self.logits.float() / self.config.temperature).softmax(-1).to(final.dtype)
        normalized = final.float() * torch.rsqrt(final.float().square().mean(-1, keepdim=True) + self.eps)
        logits = self.router(normalized.to(final.dtype))
        return (logits.float() / self.config.temperature).softmax(-1).to(final.dtype)

    def conditional_weights(self, states):
        if self.config.mode == 'learned_kv':
            # Six global learned weights, shared across tokens and examples.
            weights = (self.logits.float() / self.config.temperature).softmax(-1)
            return weights.expand(*states[-1].shape[:-1], -1)
        scores = []
        for state in states:
            x = state.float()
            normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            scores.append((normalized * self.score.float()).sum(-1) / math.sqrt(x.shape[-1]))
        logits = torch.stack(scores, dim=-1) + self.layer_bias.float()
        return (logits / self.config.temperature).softmax(-1)

    def forward(self, states):
        if len(states) != len(self.config.source_layers):
            raise ValueError('Missing fusion source states')
        final = states[-1]
        if self.config.mode == 'last':
            return final
        weights = self.routing_weights(final)
        # Avoid an additional [B,T,K,d] stacked allocation.
        result = states[0] * weights[..., 0, None]
        for i, state in enumerate(states[1:], 1):
            result = result + state * weights[..., i, None]
        return result


class MemoryFusionYOCO(YOCO):
    def __init__(self, args, fusion, **kwargs):
        super().__init__(args, **kwargs)
        self.memory_fusion = DepthMemoryFusion(args.dim, args.norm_eps, fusion)

    def forward(self, x, start_pos=0, incremental_state=None,
                is_prefilling=True, skip_cross_decoder=False):
        if self.memory_fusion.config.mode == 'last':
            return super().forward(x, start_pos, incremental_state, is_prefilling, skip_cross_decoder)
        x = self.embed_scale * self.embed_tokens(x)
        final, states = self.self_decoder(
            x, incremental_state=incremental_state, start_pos=start_pos,
            is_prefilling=is_prefilling,
            return_layers=self.memory_fusion.config.source_layers,
        )
        if self.memory_fusion.config.mode in {'learned_kv', 'conditional_kv'}:
            weights = self.memory_fusion.conditional_weights(states)
            x = self.cross_decoder(final, start_pos=start_pos,
                                   incremental_state=incremental_state,
                                   skip_cross_decoder=skip_cross_decoder,
                                   memory_sources=states, memory_weights=weights)
            return self.output_layer(self.layer_norm(x)), None
        memory = self.memory_fusion(states)
        # Query stream remains the original final self-decoder output.
        x = self.cross_decoder(final, start_pos=start_pos,
                               incremental_state=incremental_state,
                               skip_cross_decoder=skip_cross_decoder, memory=memory)
        return self.output_layer(self.layer_norm(x)), None
