"""
Sentinel Tokenizer — custom Byte-Pair Encoding (BPE) tokenizer.

Builds vocabulary from training corpus, encodes/decodes text,
and handles padding / truncation for batch inference.

No external tokenizer libraries required.
"""

import re
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
}

# ─────────────────────────────────────────────
# Normaliser / pre-tokeniser
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, keep alphanumeric + basic punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _pre_tokenise(text: str) -> List[str]:
    """Split into words (preserving apostrophes, slashes, dots in paths)."""
    return re.findall(r"[a-z0-9]+(?:['.\/\-][a-z0-9]+)*|[^\w\s]", text)


# ─────────────────────────────────────────────
# BPE helpers
# ─────────────────────────────────────────────

def _get_pairs(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    """Count all adjacent symbol pairs in the vocabulary."""
    pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _merge(vocab: Dict[Tuple[str, ...], int], pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
    """Merge the best pair across the vocabulary."""
    new_vocab: Dict[Tuple[str, ...], int] = {}
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


# ─────────────────────────────────────────────
# Tokenizer class
# ─────────────────────────────────────────────

class SentinelTokenizer:
    """
    Custom BPE tokenizer.

    Build:   tokenizer.train(corpus_sentences, vocab_size=8192)
    Save:    tokenizer.save(path)
    Load:    SentinelTokenizer.load(path)
    Use:     ids = tokenizer.encode("open chrome")
    """

    def __init__(self):
        self.vocab:    Dict[str, int] = dict(SPECIAL_TOKENS)      # token → id
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges:   List[Tuple[str, str]] = []
        self.max_seq_len: int = 128

    # ── Training ────────────────────────────────────────────────────

    def train(self, sentences: List[str], vocab_size: int = 8192, verbose: bool = True) -> None:
        """Train BPE on a list of sentences until vocab_size is reached."""
        if verbose:
            print(f"[Tokenizer] Training on {len(sentences)} sentences …")

        # Build initial character-level vocabulary
        word_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for sent in sentences:
            for word in _pre_tokenise(_normalise(sent)):
                # Represent word as tuple of characters + end-of-word marker
                chars = tuple(list(word) + ["</w>"])
                word_freq[chars] += 1

        # Collect initial character symbols
        char_set = set()
        for word in word_freq:
            char_set.update(word)

        # Seed vocabulary with special tokens + characters
        for ch in sorted(char_set):
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inv_vocab[idx] = ch

        # BPE merge loop
        num_merges = vocab_size - len(self.vocab)
        for i in range(max(0, num_merges)):
            pairs = _get_pairs(word_freq)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            word_freq = _merge(word_freq, best)
            merged_token = "".join(best)
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.inv_vocab[idx] = merged_token
            self.merges.append(best)

            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i+1}/{num_merges}  vocab={len(self.vocab)}")

        if verbose:
            print(f"[Tokenizer] Done. Vocab size: {len(self.vocab)}")

    # ── Encoding ────────────────────────────────────────────────────

    def _bpe_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) + ["</w>"]
        for merge in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1
                        and symbols[i] == merge[0]
                        and symbols[i + 1] == merge[1]):
                    new_symbols.append("".join(merge))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_mask: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[bool]]]:
        """
        Encode text to token IDs.

        Returns list of ints, optionally padded to max_length.
        """
        max_length = max_length or self.max_seq_len

        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.vocab["<CLS>"])

        for word in _pre_tokenise(_normalise(text)):
            for sym in self._bpe_word(word):
                tokens.append(self.vocab.get(sym, self.vocab["<UNK>"]))

        if add_special_tokens:
            tokens.append(self.vocab["<SEP>"])

        # Truncate
        tokens = tokens[:max_length]

        if not padding:
            if return_mask:
                mask = [False] * len(tokens)
                return tokens, mask
            return tokens

        # Pad
        pad_id = self.vocab["<PAD>"]
        length = len(tokens)
        tokens += [pad_id] * (max_length - length)

        if return_mask:
            # True = padding position
            mask = [False] * length + [True] * (max_length - length)
            return tokens, mask

        return tokens

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ):
        """
        Encode a batch of texts, returning padded tensors.

        Returns:
            input_ids     (B, T) as list-of-lists
            padding_masks (B, T) as list-of-lists  True = padding
        """
        max_length = max_length or self.max_seq_len
        encoded = [self.encode(t, max_length=max_length, padding=True, return_mask=True)
                   for t in texts]
        ids   = [e[0] for e in encoded]
        masks = [e[1] for e in encoded]
        return ids, masks

    # ── Decoding ────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        special = set(SPECIAL_TOKENS.values())
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            tok = self.inv_vocab.get(i, "<UNK>")
            tokens.append(tok)

        # Merge </w> markers back into spaces
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        return text

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save vocab + merges to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab":   self.vocab,
            "merges":  self.merges,
            "max_seq_len": self.max_seq_len,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SentinelTokenizer":
        """Load a previously trained tokenizer."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab    = {k: int(v) for k, v in data["vocab"].items()}
        tok.inv_vocab = {int(v): k for k, v in data["vocab"].items()}
        tok.merges   = [tuple(m) for m in data["merges"]]
        tok.max_seq_len = data.get("max_seq_len", 128)
        return tok

    # ── Convenience ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.vocab)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"SentinelTokenizer(vocab_size={len(self.vocab)}, merges={len(self.merges)})"
