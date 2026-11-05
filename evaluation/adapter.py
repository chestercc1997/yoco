"""Teacher-forced lm-evaluation-harness adapter for either architecture."""

import json
from pathlib import Path

import torch
from lm_eval.api.model import TemplateLM
from tqdm import tqdm

from runtime.model import build_model, tokenizer
from runtime.precision import configure as configure_precision, autocast


class ReproductionLM(TemplateLM):
    def __init__(self, config, checkpoint, device="cuda:0"):
        super().__init__()
        self.config = json.loads(Path(config).read_text())
        self.precision = self.config.get('evaluation_precision', self.config['training'].get('precision','fp16_mixed_fp32_params'))
        configure_precision(self.precision)
        if 'allow_tf32' in self.config.get('execution',{}):
            torch.backends.cuda.matmul.allow_tf32=self.config['execution']['allow_tf32']
            torch.backends.cudnn.allow_tf32=self.config['execution']['allow_tf32']
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(2)
        self.tokenizer = tokenizer()
        self.model = build_model(self.config).to(self.device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.model.load_state_dict(saved["model"])
        self.step = saved.get("step", 0)
        self.model.eval()
        self.max_length = self.config["model"]["max_seq_len"]

    @property
    def eot_token_id(self): return self.tokenizer.eos_id

    @property
    def tokenizer_name(self): return "official_yoco_cl100k_base"

    def tok_encode(self, string, **kwargs): return self.tokenizer.encode(string)

    @torch.inference_mode()
    def score_tokens(self, context, continuation):
        if not continuation:
            return 0.0, True
        context = context or [self.eot_token_id]
        sequence = context + continuation
        total, greedy, start = 0.0, True, len(context)
        while start < len(sequence):
            end = min(len(sequence), start + self.max_length)
            left = max(0, end - self.max_length - 1)
            if left >= start: left = start - 1
            tokens = torch.tensor(sequence[left:end - 1], device=self.device)[None, :]
            with autocast(self.precision):
                logits = self.model(tokens)[0]
            selected = logits[0, start-left-1:end-left-1].float()
            targets = torch.tensor(sequence[start:end], device=self.device)
            total += (-torch.nn.functional.cross_entropy(selected, targets, reduction="sum")).item()
            greedy = greedy and bool(torch.equal(selected.argmax(-1), targets))
            start = end
        return total, greedy

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        results = []
        for key, context, continuation in tqdm(requests, disable=disable_tqdm, desc="Scoring"):
            result = self.score_tokens(context, continuation)
            results.append(result)
            if key is not None: self.cache_hook.add_partial("loglikelihood", key, result)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        return [self.score_tokens([self.eot_token_id], self.tok_encode(request.args[0]))[0] for request in requests]

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("This recipe evaluates multiple-choice scoring tasks")
