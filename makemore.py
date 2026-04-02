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

# RNN Language Model and GRU Language Model

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
        cell_type = cell_type.lower()
        if cell_type == 'rnn':
            self.rnn_cell = RNNcell(config) # RNN cell for processing the sequence
        elif cell_type == 'gru':
            self.rnn_cell = GRUcell(config) # GRU cell for processing the sequence
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Expected 'rnn' or 'gru'.")
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

# MLP Language Model

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates them together, and then applies an MLP to the result and uses that to predict the next token in the sequence.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embedding table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        # MLP to process the concatenated embeddings of the previous tokens
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd * config.block_size, config.n_embd2), # first layer of the MLP, projects from block_size*n_embd to n_embd2
            nn.Tanh(), # activation function
            nn.Linear(config.n_embd2, config.vocab_size), # second layer of the
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # gather the words embedding of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # (B, T, n_embd)
            idx = torch.roll(idx, shifts=1, dims=1) # shift the input to the right by one position
            idx[:, 0] = self.vocab_size # set the first position to the <BLANK> token
            embs.append(tok_emb)

        # concatenate all of the embeddings together and pass through the MLP
        x = torch.cat(embs, dim=-1) # (B, T, block_size*n_embd)
        logits = self.mlp(x) # (B, T, vocab_size)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ((B*T, vocab_size), (B*T))
        
        return logits, loss

#---------------------------------------

# Bigram Language Model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
# -----------------------------------------------------------------------------

# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------

# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def create_datasets(input_file):

    # Allow running from outside the makemore folder by resolving relative paths
    # against the directory that contains this script.
    if not os.path.isabs(input_file) and not os.path.exists(input_file):
        input_file = os.path.join(os.path.dirname(__file__), input_file)

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    # Always persist the latest model at the end of training, even when no
    # evaluation checkpoint has been triggered yet.
    final_out_path = os.path.join(args.work_dir, "model.pt")
    torch.save(model.state_dict(), final_out_path)
    print(f"saved final model to {final_out_path}")