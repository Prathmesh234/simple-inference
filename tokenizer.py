"""
Tokenizer: wrap HuggingFace's tokenizer for Llama 3.

What is a tokenizer?
---------------------
Text is a string of characters. Models operate on integers (token IDs).
The tokenizer is the bridge: it converts text → list[int] (encode)
and list[int] → text (decode).

Llama 3 uses Byte-Pair Encoding (BPE) via the tiktoken library.
BPE works by:
  1. Start with a vocabulary of individual bytes (256 entries)
  2. Repeatedly find the most common adjacent pair of tokens in the training
     corpus and merge them into a new token
  3. Repeat until vocab_size is reached (128,256 for Llama 3)

The result: common words become single tokens ("the" → [279]),
rare words split into subword pieces ("inference" → [258, 11862]).
This lets the model handle any text while keeping the vocabulary finite.

Special tokens
--------------
Llama 3 has special tokens that the model was trained to recognize:
  <|begin_of_text|>  (id=128000) — always prepended to a sequence
  <|end_of_text|>    (id=128001) — signals the model to stop generating
  <|eot_id|>         (id=128009) — end of a chat turn

Why BOS matters: the model was trained with BOS at the start of every
sequence. If you forget it, the model sees an out-of-distribution input
and output quality degrades.

Why the vocab is 128,256 and not a round number:
  128,000 BPE tokens + 256 reserved special token slots = 128,256.
  Most special slots are unused but kept for future fine-tuning.
"""

from __future__ import annotations

import env_loader
from pathlib import Path
from typing import List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizerFast


class Tokenizer:
    """
    Thin wrapper around HuggingFace's PreTrainedTokenizerFast.

    We wrap rather than subclass so we can expose only what we need
    and add pedagogical helpers alongside.
    """

    def __init__(self, model_dir: str | Path):
        """Load tokenizer from a local model directory."""
        self._tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            str(model_dir),
            use_fast=True,
        )

    @classmethod
    def from_pretrained(cls, repo_id: str, token: Optional[str] = None) -> Tokenizer:
        """Load tokenizer directly from HuggingFace Hub (no weights needed)."""
        import os
        token = token or os.environ.get("HF_TOKEN")
        obj = object.__new__(cls)
        obj._tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True, token=token)
        return obj

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """
        Convert text to a list of token IDs.

        add_bos=True prepends the <|begin_of_text|> token (id=128000).
        You almost always want this for the first (and only) sequence
        you feed to the model.
        """
        ids = self._tok.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Convert a list of token IDs back to a string.

        skip_special=True hides tokens like <|begin_of_text|> from output.
        Set it to False when debugging to see every token explicitly.
        """
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str], add_bos: bool = True) -> List[List[int]]:
        """Encode multiple strings."""
        return [self.encode(t, add_bos=add_bos) for t in texts]

    # ------------------------------------------------------------------
    # Token-level inspection helpers (good for learning)
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """Return the list of token strings (not IDs) for a piece of text."""
        return self._tok.tokenize(text)

    def show_tokens(self, text: str, add_bos: bool = True):
        """
        Print each token alongside its ID — very useful for understanding
        how the model sees your input.

        Example output:
            [ 0]  128000  '<|begin_of_text|>'
            [ 1]     791  'The'
            [ 2]    4087  ' quick'
            ...
        """
        ids = self.encode(text, add_bos=add_bos)
        print(f"  {'pos':>5}  {'id':>8}  token")
        print(f"  {'---':>5}  {'--':>8}  -----")
        for pos, tid in enumerate(ids):
            # decode single token, keep specials visible
            tok_str = self._tok.decode([tid], skip_special_tokens=False)
            tok_str = repr(tok_str)  # show whitespace and escape chars
            print(f"  {pos:>5}  {tid:>8}  {tok_str}")
        print(f"\n  Total tokens: {len(ids)}")

    def id_to_token(self, token_id: int) -> str:
        """Look up the string for a single token ID."""
        return self._tok.decode([token_id], skip_special_tokens=False)

    def token_to_id(self, token_str: str) -> int:
        """Look up the ID for a token string (exact match)."""
        return self._tok.convert_tokens_to_ids(token_str)

    # ------------------------------------------------------------------
    # Special token properties
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        return self._tok.bos_token_id  # 128000

    @property
    def eos_id(self) -> int:
        return self._tok.eos_token_id  # 128001

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size    # 128256

    # ------------------------------------------------------------------
    # Info / diagnostics
    # ------------------------------------------------------------------

    def print_summary(self):
        print("=" * 52)
        print("  Tokenizer")
        print("=" * 52)
        print(f"  Class       : {type(self._tok).__name__}")
        print(f"  Vocab size  : {self.vocab_size:,}")
        print(f"  BOS token   : '{self.id_to_token(self.bos_id)}'  (id={self.bos_id})")
        print(f"  EOS token   : '{self.id_to_token(self.eos_id)}'  (id={self.eos_id})")

        # show a few other special tokens
        special = self._tok.added_tokens_encoder
        print(f"  Special tokens added: {len(special)}")
        shown = 0
        for tok_str, tok_id in sorted(special.items(), key=lambda x: x[1]):
            if shown >= 6:
                print(f"    ... ({len(special) - shown} more)")
                break
            print(f"    {tok_id:>8}  {repr(tok_str)}")
            shown += 1
        print("=" * 52)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Section 3 — Tokenizer")
    print(f"{'='*60}\n")

    tok = Tokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tok.print_summary()

    # --- round-trip test ---
    print("\n--- Round-trip test ---")
    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def forward(self, x: torch.Tensor) -> torch.Tensor:",
        "inference",        # splits into subwords
        "supercalifragilisticexpialidocious",
    ]
    all_pass = True
    for text in samples:
        ids  = tok.encode(text, add_bos=False)
        back = tok.decode(ids)
        ok   = back == text
        all_pass = all_pass and ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {repr(text)[:45]:<47} → {len(ids)} tokens → {repr(back)[:45]}")

    print(f"\n  Round-trip: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # --- token inspection ---
    print("\n--- Token breakdown: 'inference engine' ---")
    tok.show_tokens("inference engine", add_bos=False)

    print("\n--- Token breakdown: full prompt with BOS ---")
    tok.show_tokens("Simply put, the theory of relativity states that", add_bos=True)

    # --- compression ratio ---
    print("\n--- Compression ratio ---")
    test_text = (
        "Large language models are neural networks trained on massive text corpora. "
        "They learn to predict the next token given a context window of previous tokens. "
        "At inference time, tokens are generated one at a time in an autoregressive loop."
    )
    ids = tok.encode(test_text, add_bos=False)
    chars_per_token = len(test_text) / len(ids)
    print(f"  Text  : {len(test_text)} characters")
    print(f"  Tokens: {len(ids)}")
    print(f"  Ratio : {chars_per_token:.2f} chars/token")
    print(f"  (English averages ~4 chars/token with BPE)")
    print("\nDone.")
