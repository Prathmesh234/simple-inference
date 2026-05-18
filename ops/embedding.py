"""
TokenEmbedding and OutputProjection.

What embeddings do
------------------
The model operates on continuous vectors, not integers. The embedding table
is a lookup: given a token ID, return the corresponding row of a learned
(vocab_size, hidden_size) weight matrix. That row is the token's initial
representation before any transformer layers run.

    token_id = 279   ("the")
    embedding = weight[279]   # shape: (hidden_size,)

For a sequence of T tokens you get shape (T, hidden_size), which becomes
the input to the first transformer block.

Tied embeddings
---------------
The output projection (the final step that converts hidden states back to
logit scores over the vocabulary) reuses the exact same weight matrix as
the input embedding. This is called weight tying.

    Input:  token_id → weight[token_id]            (lookup, shape: hidden_size)
    Output: hidden   → hidden @ weight.T            (matmul, shape: vocab_size)

Why tie them?
  - The embedding row for a token represents "what this token means as input"
  - The output projection row for a token represents "how likely is this token next"
  - Tying forces these two representations to live in the same space, which
    works well in practice and saves ~1.2 GB of parameters for this model
    (vocab_size × hidden_size × 2 bytes = 128256 × 3072 × 2 = 787 MB × 2 without tying)

Shape contract
--------------
TokenEmbedding:
  input:  (batch, seq_len)          integer token IDs
  output: (batch, seq_len, hidden_size)

OutputProjection:
  input:  (batch, seq_len, hidden_size)
  output: (batch, seq_len, vocab_size)   these are raw logits, not probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # Standard nn.Embedding: lookup table of shape (vocab_size, hidden_size)
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) integers in [0, vocab_size)
        # returns:   (B, T, hidden_size)
        return F.embedding(token_ids, self.weight)

    def load_weight(self, weight: torch.Tensor):
        ##this is very important. - here is where the model loading happens 
        ## basically without tracking the gradients we launch the model copying the weights 
        ## from the disk to the gpu into the embed token class
        with torch.no_grad():
            self.weight.copy_(weight)


class OutputProjection(nn.Module):
    """
    The final linear layer: hidden states → vocabulary logits.

    Does NOT own its weight — it holds a reference to TokenEmbedding's weight.
    This is the tied-embedding contract: one tensor, two uses.
    """

    def __init__(self, embedding: TokenEmbedding):
        super().__init__()
        # Register as a buffer reference so it moves with .to(device) calls
        # but is NOT counted as a separate parameter
        self._embedding = embedding

    @property
    def weight(self) -> torch.Tensor:
        return self._embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, hidden_size)
        # weight: (vocab_size, hidden_size)
        # output: (B, T, vocab_size)
        return F.linear(x, self.weight)  # equivalent to x @ weight.T
