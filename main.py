import os
import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
from matplotlib import pyplot as plt

### step1. input block ###

# select avaiable device 
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data file
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r') as f:
    data = f.read()

# all the unqiue chars from the input.txt
vocab = sorted(list(set(data)))
print(f"\nvocab of all the sorted unique chars: {vocab}\n")

# training llama model requires such as <|begin_of_text|>, <|end_of_text|>, <|pad_id|>, so add them into vocab
vocab.extend(['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>'])
vocab_size: int = len(vocab)
print(f"vocab_size: {vocab_size:,}\n")

# create a mapping between chars with corresponding intergers indexes in vocab
itos = { i : ch for i, ch in enumerate(vocab) }
stoi = { ch : i for i, ch in enumerate(vocab) }
print(f"integers to string: {itos}\n\n string to integers: {stoi}\n")

# func encode and decode 
def encode(s): # take a string output a list of integers
    return [stoi[ch] for ch in s] # stoi need a char so pass a string(s) to enumerate each char 

def decode(l): # take a list of int output a string
    return ''.join(itos[i] for i in l) # itos need a index(i) so pass a list to enumerate each integer

# define tensor token var to be used later during model training
token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad_id|>']], dtype=torch.int, device=device)

prompts = "Hello, World."
encode_tokens = encode(prompts) # str -> int
print(f"encode_tokens: {encode_tokens}\n\n")
decode_text = decode(encode_tokens) # int -> str
print(f"decode_text: {decode_text}\n\n")

### step2. the decoder block ###
# Root Mean Square(1/n(the square of the first root) + the square of the second root + the square of the nth root)

@dataclass
class ModelArgs:
    dim: int = 512      # embedding dimension
    n_layers: int = 8       # number of model decoder block
    n_heads: int = 8        # number of heads for queires embedding
    n_kv_heads: int = 4     # number of heads for keys and values embedding
    #vocab_size: int = len(vocab)       # length of vocabulary(has been defined above)
    multiple_of: int = 256      # require to calculate dim of feedforward network
    ffn_dim_multiplier: Optional[float] = None      # require to calculate dim of feedforward network
    norm_eps: float = 1e-5      # default epsilon value set for the RMSNorm calculation
    rope_theta: float = 10000.0     # default theta bvalue for the RePE calculation
    max_bach_size: int = 10
    max_seq_len: int = 256 
    epochs: int = 2500      # total number of training iteration
    log_interval: int = 10      # number of interval to print logs and loss value
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

## a. the RMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        device = ModelArgs.device
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim).to(device))      # scaling parameter gamma, initialized with one and the number of parameters is equal to the size of dim
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(device)
    
    def forward(self, x):
        # shape: x[bs, seq, dim]

        output = self._norm(x.float()).type_as(x) # x_norm0
        # shape: x[bs, seq, dim] -> x_norm[bs, seq, dim]
        return output * self.weight     # x_norm1(x * y)

## test: RMSNorm code
x = torch.randn(ModelArgs.max_bach_size, ModelArgs.max_seq_len, ModelArgs.dim, device=device)
rms_norm = RMSNorm(dim=ModelArgs.dim)
x_norm = rms_norm(x)

print(f"shape of x: {x.shape}\n\n")
print(f"shape of x_norm: {x_norm.shape}\n\n")

## b. Rotary Positional Encoding(RoPE) -> define thoe position of the each token in the sentences

# the RoPE
def precompute_freqs_cis(dim: int, seq_len: int, theta: float=10000.0):
    # computing theta value for each dim pair which is dim/2
    device = ModelArgs.device
    # init freqs
    freqs = 1.0 /(theta ** (torch.arange(0, dim, 2, device=device)[:(dim//2)].float()/dim))

    # computing range of positions(m) in the sequence
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # freqs gives all the theta value range for all the position of tokens in the sequence
    freqs = torch.outer(t, freqs).to(device) # torch.outer: computes the outer product of two 1D tensors

    # rotation matrix which is converted to polar form in order to perform rotation to the embedding
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
    return freqs_cis

# reshape outer after freqs_cis ?
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "the last two dimension of freqs_cis, x must match"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# start rotation
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = ModelArgs.device
    # applying RoPE to both query and key embedding together
    # First, the last dimension of xq, xk embedding needs to be reshaped to make it a pair. as rotation matrix is applied to each pari of dim.
    # # Next: convert both xq and xk to complex number , as the rotation matrix is only applicable to complex number
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device) # xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device) # xk_:[bsz, seq_len, n_heads, head_dim/2]

    # The rotation matrix(freqs_cis) dimensions across seq_len(dim=1) and head_dim(dim=3) should match with the embedding
    # Also, the shape freqs_cis should be the same with xq and xk, hence change the shape of freqs_cis:[seq_len,head_dim] -> freqs_cis:[1,seq_len,1,head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Finally, perform rotation operation by multiplying with freqs_cis
    # after rotation, convert out of xq and xk back to real number and return 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device) # xq_out: [bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device) # xk_out: [bsz, seq_len, n_heads, head_dim]
    return xq_out.type_as(xq), xk_out.type_as(xk)

## test: RoPE code
# by having multiple heads attention, model can learn different attention patterns simultaneously
head_dim = ModelArgs.dim//ModelArgs.n_heads # head_dim == total of model.dim is divided by the number of heads
wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device) # layer wq: transform the input into query vectors for each attention head
wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)# layer wk: transform the input into key vector
# use wq and wk layer, allow the model focus on different parts of the input data
xq = wq(x_norm)
xk = wk(x_norm)
print(f"xq.shape: {xq.shape}\n")
print(f"xk.shape: {xk.shape}\n")

# convert to complex number
xq = xq.view(xq.shape[0],xq.shape[1],ModelArgs.n_heads, head_dim)
xk = xk.view(xk.shape[0],xk.shape[1],ModelArgs.n_kv_heads, head_dim)
print(f"xq.re-shape: {xq.shape}\n")
print(f"xk.re-shape: {xk.shape}\n")

# theta value 
freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)
print(f"freqs_cis.shape: {freqs_cis.shape}\n")

xq_rotate, xk_rotate = apply_rotary_emb(xq, xk, freqs_cis)
print(f"xq_rotate.shape: {xq_rotate.shape}\n")
print(f"xk_rotate.shape: {xk_rotate.shape}\n\n")

## c: KV Cache(required at inference) + d: Group Query Attention

## the input data is first normalized by a. RMS (reducing the size of each x, y), and then the multi-head attention mechanism is used for the query in b. rotation matrix stage, k uses the Group Query Attention in the RoPE, and v uses the Group Quert Attention
## c. KV Cache(only required at inferencing) (store previously generated tokens in the form of key and value cache, for eample, reduces the matrix multiplication from 3*3 to 1(which has stroed the preivious tokens)*3)
## d. Gropu Query Attention(to reduce the number of heads for k and v)
## in Multi-Head Attention: the Queries num = number of heads, and values and keys = number of heads, Q, K, V(8, 8, 8)
## in Group Query Attention: Number of Query Heads(n_heads) = Queries, but keys and values = 4(2 times less than query heads), Q, K, V(8, 4, 4)
## The Attention Block [c: The KV Cache, d: Group Query Attention]

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Embedding dimension
        self.dim = args.dim
        # Number of heads assigned to Query
        self.n_heads = args.n_heads
        # heads of Key and Value, if "None, the number will be same as Query"
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # dimension of each head relative to model dim
        self.head_dim = args.dim // args.n_heads
        # Number of repetition in order to make times key, value heads to match Query heads number
        self.n_rep = args.n_heads // args.n_kv_heads

        # Weight initialize for keys, Querys, Values and output
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wo = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)

        # Initialize caches to store key, value as start (KV Cache Implementation)
        self.cache_k = torch.zeros((args.max_bach_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=device)
        self.cache_v = torch.zeros((args.max_bach_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=device)

    def forward(self, x: torch.Tensor, start_pos, inference):
        # Shape of the input embedding: [bsz, seq_len, dim]
        bsz, seq_len, _ = x.shape
        # Mask will be used during 'Training' and is not required for 'inference' due to the use of KV cache
        mask = None

        xq = self.wq(x)  # x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
        xk = self.wk(x)  # x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
        xv = self.wv(x)  # x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

        # Reshaping Querys, Keys and Values by their number of head (Group Query Attention Implementation)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      # xq[bsz,seq_len,n_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # xk[bsz,seq_len,n_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # xv[bsz,seq_len,n_kv_heads, head_dim]

        # Model - Inference Mode: KV-Cache is enabled at inference model only
        if inference:
            # Compute rotation matrix for each position in the sequence
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len = self.args.max_seq_len * 2)
            #  take the rotation matrix range from the current position of the tokens
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
            # apply RoPE to Queries and Keys embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            # store keys and values token embeddings into their respective cache[KV Cache Implementation]
            self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv
            
            # assign all the previous token embedding upto current tokens position to keys and values var for Attention Caculation
            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]

            # the keys and values shape aren't same with Queries embedding which has to be in order to compute attention score
            # so use repead_kv func to make k, v shape same as Q
            # repeat_kv = self.n_rep
            keys = repeat_kv(keys, self.n_rep) # keys[bsz, seq_len, n_heads, head_dim]
            values = repeat_kv(values, self.n_rep) # values[bsz, seq_len, n_hedas, head_dim]

        # Mode - Training mode: KV-Cache not implemented
        else: 
            # compute rotatiton matrix and apply RoPE to q and k, v for training
            freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)

            # xq[bsz, seq_len, n_heads, head_dim], xk[bsz, seq_len, n_heads, head_dim]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

            # use repeat_kv to make k, v shape same as q shape
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)

            # for training mode, we need compute mask and apply to the attention score later
            mask = torch.full((seq_len, seq_len), float("-inf"), device=self.args.device) # to be determined
            mask = torch.triu(mask, diagonal=1).to(self.args.device)

            # to compute attention, reshape q, k, v birngs heads at dim1 and seq at dim2
            xq = xq.transpose(1, 2) # xq[bsz, n_heas, seq_len, head_dim]
            keys = keys.transpose(1, 2) # ...
            values = values.transpose(1, 2) # ...

            # compute attention score
            scores = torch.matmul(xq, keys.transpose(2, 3)).to(self.args.device) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask

            # apply softmax to attention score
            scores = F.softmax(scores.float(), dim = -1).type_as(xq)
            # matrix mul of attention socre with values
            output = torch.matmul(scores, values).to(self.args.device)

            # shape change: output[bsz, n_heads, seq_len, head_dim] -> [bsz, seq_len, n_heads, head_dim] -> [bsz, seq_len, n_heads * head_dim]
            output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

            # shape: output[bsz, seq_len, dim]
            return self.wo(output)
        
# repeat func fot the k, v repetition
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,:,None,:]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)

    )
        
## test c. and d.
n_rep = ModelArgs.n_heads // ModelArgs.n_kv_heads
keys = repeat_kv(xk, n_rep) 
print(f"xk.shape: {xk.shape}\n")
print(f"keys.shape: {keys.shape}\n")
# test attention func
attention = Attention(ModelArgs)
x_out = attention(x_norm, start_pos = 0, inference = False)
print(f"x_out.shape: {x_out.shape}\n")

## e. FeedForward Network(SwiGLU Activation)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        self.dim = dim

        # use hidden_dim calculation
        hidden_dim = int(2 * hidden_dim/3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # define hidden layers weights
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

    def forward(self, x):
        # shape: [bsz, seq_len, dim]
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
## test: FeedForward module
feed_forward = FeedForward(ModelArgs.dim, 4 * ModelArgs.dim, ModelArgs.multiple_of, ModelArgs.ffn_dim_multiplier)
x_out = rms_norm(x_out)
x_out = feed_forward(x_out)
print(f"\nfeed forward output: x_out.shape: {x_out.shape}\n")

## f. Decoder Block
    


            