"""Tokenizer wrapper used by the released YOCO training code."""

from typing import List

import tiktoken


class TiktokenTokenizer:
    def __init__(self, model: str = "cl100k_base", pad_to_multiple: int = 8):
        self.symbols = ["<s>", "<pad>", "</s>", "<unk>"]
        self.indices = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.pad_to_multiple = pad_to_multiple
        base = tiktoken.get_encoding(model)
        self._model = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                "<fim_prefix>": 100264, "<fim_middle>": 100265,
                "<fim_suffix>": 100266, "<fim_pad>": 100267,
                "<reponame>": 100268, "<filename>": 100269,
                "<gh_stars>": 100270, "<issue_start>": 100271,
                "<issue_comment>": 100272, "<issue_closed>": 100273,
                "<jupyter_start>": 100274, "<jupyter_text>": 100275,
                "<jupyter_code>": 100276, "<jupyter_output>": 100277,
                "<empty_output>": 100278, "<commit_before>": 100279,
                "<commit_msg>": 100280, "<commit_after>": 100281,
            },
        )

    @property
    def n_words(self) -> int:
        size = self._model.n_vocab + len(self.symbols)
        return (size + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple

    @property
    def bos_id(self) -> int: return self.indices["<s>"]

    @property
    def eos_id(self) -> int: return self.indices["</s>"]

    @property
    def pad_id(self) -> int: return self.indices["<pad>"]

    @property
    def unk_id(self) -> int: return self.indices["<unk>"]

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        tokens = [token + len(self.symbols) for token in self._model.encode(text, allowed_special="all")]
        if bos: tokens.insert(0, self.bos_id)
        if eos: tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self._model.decode([token - len(self.symbols) for token in tokens if token >= len(self.symbols)])
