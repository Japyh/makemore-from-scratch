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

class CausalBoW(nn.Module):
    """
    Causal BoW. Averages the preceding elements and looks suspiciously like a CausalAttention
    module you'd find in a transformer, for no apparent reason at all.
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, config.block_size, config.block_size)) # (1, block_size, block_size)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # do the weighted average of the previous elements in the sequence
        att = torch.zeros(B, T, T, device=x.device) # (B, T, T)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf')) # (B, T, T)
        att = F.softmax(att, dim=-1) # (B, T, T)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)
        return y
    
class BoWBlock(nn.Module):
    """ Collects BoW features and then applies an MLP to them. """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP to process the BoW features
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(config.n_embd, config.n_embd2), # first layer of the MLP, projects from n_embd to n_embd2
            c_proj = nn.Linear(config.n_embd2, config.n_embd), # second layer of the MLP, projects back to n_embd
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward pass
    
    def forward(self, x):
        x = x + self.cbow(x) # BoW with residual connection
        x = x + self.mlpf(x) # MLP with residual connection
        return x

class BoW(nn.Module):
    """
    Takes the previous block_size tokens, encodes them with a lookup table, 
    also encodes their positions with lookup table, averages them together, and then applies an MLP to the result.
    and uses that to predict the next token in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embedding table
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd) # position embedding table
        # context block
        self.context_block = BoWBlock(config) # context block to average the previous elements in the sequence
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) # language modeling head

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

        # forward the BoW model itself
        token_emb = self.wte(idx) # token embeddings (B, T, n_embd)
        position_emb = self.wpe(pos) # position embeddings (1, T, n_embd)
        # add and run through the decoder MLP to get the logits for the next token in the sequence
        x = token_emb + position_emb # (B, T, n_embd)
        # run the bag of words context module
        x = self.context_block(x) # (B, T, n_embd)
        # decode to next token probability
        logits = self.lm_head(x) # (B, T, vocab_size)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ((B*T, vocab_size), (B*T))

        return logits, loss
    
# ---------------------------------------

"""
Recurrent Neural Network (RNN) language model: either a vanilla RNN reccurence or a GRU.
Did not implement an LSTM because its API is a bit moreannoying it has both a hidden state and a cell state,
but it's very similar to GRU and in practice works just as well.
"""

class RNNcell(nn.Module):
    """
    the job of a "Cell" is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state h_{t} for the current time step.
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2) # linear layer to combine input and hidden state
        
    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1) # concatenate input and hidden state (B, n_embd + n_embd2)
        ht = F.tanh(self.xh_to_h(xh)) # apply linear layer and nonlinearity to get the next hidden state (B, n_embd2)
        return ht
        
class GRUcell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2) # linear layer to compute update gate
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2) # linear layer to compute reset gate
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2) # linear layer to compute candidate hidden state
        
    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hiddden state to zero
        xh = torch.cat([xt, hprev], dim=1) # concatenate input and hidden state (B, n_embd + n_embd2)
        r = F.sigmoid(self.xh_to_r(xh)) # reset gate (B, n_embd2)
        hprev_reset = r * hprev # apply reset gate to hidden state (B, n_embd2)
        # calculate the candidate new hidden state using the reset hidden state
        xhr = torch.cat([xt, hprev_reset], dim=1) # concatenate input and reset hidden state (B, n_embd + n_embd2)
        hbar = F.tanh(self.xh_to_hbar(xhr)) # candidate hidden state (B, n_embd2)
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh)) # update gate (B, n_embd2)
        # combine the previous hidden state and the candidate hidden state according to the update gate
        ht = (1 -  z) * hprev + z * hbar # new hidden state (B, n_embd2)
        return ht

class RNN(nn.Module):
    """
    RNN language model. Takes the previous block_size tokens, encodes them with a lookup table, 
    adds them together, and then applies an RNN recurrence to the result and uses that to predict the next token in the sequence.
    """
    def __init__(self, config, cell_type='RNN'):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # starting hidden state (1, n_embd2)
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embedding table
        # context block
        if cell_type == 'RNN':
            self.rnn_cell = RNNcell(config) # RNN cell for processing the sequence
        elif cell_type == 'GRU':
            self.rnn_cell = GRUcell(config) # GRU cell for processing the sequence
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd2, config.vocab_size) # language modeling head

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (B, T, n_embd)

        # sequentially iterate over the input and update the RNN state at each time step
        hprev = self.start.expand(B, -1) # (B, n_embd2)
        hiddens = []
        for i in range(T):
            xt = emb[:, i, :] # (B, n_embd)
            ht = self.rnn_cell(xt, hprev) # (B, n_embd2)
            hprev = ht
            hiddens.append(ht)
        
        # decode the outputs
        hidden = torch.stack(hiddens, dim=1) # (B, T, n_embd2)
        logits = self.lm_head(hidden) # (B, T, vocab_size)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ((B*T, vocab_size), (B*T))
        
        return logits, loss

# ---------------------------------------

