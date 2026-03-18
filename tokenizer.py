import re
import torch


def tokenize(text: str):
    """Split on punctuation, keeping angle-bracket tokens intact."""
    return re.findall(r"<[^>]+>|\w+|[^\w\s]", text)


def normalize_vars(text: str) -> str:
    """Replace hypothesis names (h, h1, h2, h_1, ...) with a generic token."""
    return re.sub(r"\bh\d*\b", "<HYP>", text)


class Vocab:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<HYP>": 2}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<HYP>"}
        self.next_id = 3

    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

    def encode(self, tokens):
        return [self.token_to_id.get(t, 1) for t in tokens]


def encode_text(text, vocab):
    text = normalize_vars(text)
    tokens = tokenize(text)
    return torch.tensor(vocab.encode(tokens), dtype=torch.long)
