"""
You give this script some words and it will generate more things like it.
Uses super state of the art Transformer language model to do so.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the model."""
    block_size: int = None # context length
    vocab_size: int = None # size of the vocabulary
    # parameters below control the size of the model
    n_layer: int = 4 # number of layers
    n_embd: int = 64 # embedding dimension
    n_embd2: int = 64 # second embedding dimension
    n_head: int = 4 # number of heads

# --------------------------------------
# Transformer Language Model (as used in GPT-2)

class NewGELU(nn.Module):
    """Implementation of the GELU activation function (identical to OpenAI GPT-2).
       Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415 
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class CausalSelfAttention(nn.Module):
    """A multi-head masked self-attention layer with a projection at the end.
       It is possible to use torch.nn.MultiheadAttention here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # this will be split into three parts for query, key, and value
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # (1, 1, block_size, block_size)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, n_embd) each
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)

        # causal self-attention; Self-attend: (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # (B, n_head, T, T)
        att = F.softmax(att, dim=-1) # (B, n_head, T, T)
        y = att @ v # (B, n_head, T, T) x (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side: (B, T, n_embd)

        # output projection
        y = self.c_proj(y) # (B, T, n_embd)
        return y
    
class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embd, 4 * config.n_embd), # first layer of the MLP, expands the embedding dimension to 4 times n_embd
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd), # second layer of the MLP, projects back to n_embd
            act = NewGELU() # activation function
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward pass

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # attention with residual connection
        x = x + self.mlpf(self.ln_2(x)) # MLP with residual connection
        return x

class Transformer(nn.Module):
    """Transformer Language Model, exactly as seen in GPT-2"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding table
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding table
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language modeling head

        # report number of parameters (we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

        # forward the GPT model itself
        token_emb = self.transformer.wte(idx) # token embeddings (B, T, n_embd)
        position_emb = self.transformer.wpe(pos) # position embeddings (1, T, n_embd)
        x = token_emb + position_emb # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x) # (B, T, n_embd)
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ((B*T, vocab_size), (B*T))
        
        return logits, loss

# ---------------------------------------
# Bag of Words (BoW) Language Model