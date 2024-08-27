import time
import math
import numpy as np
import torch
import inspect
import spacy
import numpy as np
import tiktoken
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset # huggingface datasets
from dataclasses import dataclass
from contextlib import nullcontext


# Load pre-trained English language model from spaCy
nlp = spacy.load('en_core_web_sm')

"""## Load Dataset | Split and Tokenize"""

# number of workers, cpu cores // 2
num_proc = 12

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

load_data = True

if load_data:
  # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
  dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

  # owt by default only contains the 'train' split, so create a test split
  split_dataset = dataset["train"].train_test_split(test_size=0.001, seed=1342, shuffle=True)
  split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

  # this results in:
  # >>> split_dataset
  # DatasetDict({
  #     train: Dataset({
  #         features: ['text'],
  #         num_rows: 8009762
  #     })
  #     val: Dataset({
  #         features: ['text'],
  #         num_rows: 4007
  #     })
  # })

  tokenize = False

  if tokenize:

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        enc = tiktoken.get_encoding(encoding_name="gpt2")
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe

        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out


    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        _file_path = '/content/drive/MyDrive/Datasets/openwebtext'
        _file_name = f'{split}.bin'
        _file_loc = f'{_file_path}/{_file_name}'
        dtype = np.uint16 # (can do since enc.max_token_value == 100000 is < 2**16)
        arr = np.memmap(_file_loc, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {_file_loc}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')

"""## Classes

### Normalization
"""

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, n_embd, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input):
        normalized_shape = input.shape[-self.weight.dim():]
        return F.layer_norm(input, normalized_shape, self.weight, self.bias, 1e-5)

"""### Phased Self-Attention

"""

class PhasedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mlp_switch = nn.Sequential(
            nn.Linear(2 * self.config.n_embd, self.config.n_embd),
            nn.GELU(),
            nn.Linear(self.config.n_embd, 2 * self.config.n_embd),
            nn.Softmax(dim=-1)
        )

        self.ff = nn.Sequential(
            nn.Linear(self.config.n_embd, 1 * self.config.n_embd, bias=self.config.bias),
            nn.GELU(),
            nn.Linear(1 * self.config.n_embd, self.config.n_embd, bias=self.config.bias),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, token_emb, pos_emb):
        # token_emb and pos_emb are embeddings for tokens and positions respectively
        # They are assumed to be of the same shape: (batch_size, seq_len, emb_dim)
        print(f"==== Phased Att Fwd ==== Input Sizes: TokenEmb: {token_emb.size()} :: PosEmb: {pos_emb.size()} ")
        # Combine token and position embeddings along the last dimension
        combined_emb = torch.cat([token_emb, pos_emb], dim=-1)
        print(f"==== Phased Att Fwd ==== Torch Concat Token - Pos: {combined_emb.size()} ")
        # Apply MLP switch to combined embeddings to obtain weights
        weights = self.mlp_switch(combined_emb)
        print(f"==== Phased Att Fwd ==== MLP Switch Output: {weights.size()}")

        # Separate weights for tokens and positions
        token_weight, pos_weight = weights.chunk(2, dim=-1)
        print(f"==== Phased Att Fwd ==== Weights :: Token Weight: {token_weight.size()} :: Position Weight: {pos_weight.size()}")

        # Apply the phase shift to both embeddings
        token_emb_shifted = self.phase_shift(token_emb, token_weight)
        print(f"==== Phased Att Fwd ==== Shifted Token Emb: {token_emb_shifted.size()}")
        pos_emb_shifted = self.phase_shift(pos_emb, pos_weight)
        print(f"==== Phased Att Fwd ==== Shifted Position Emb: {pos_emb_shifted.size()}")

        # Combine the shifted embeddings
        x = token_emb_shifted + pos_emb_shifted
        print(f'==== Phased Att ==== Combined Shifted Block Output: {x.size()}')

        return x

    def phase_shift(self, x, weight):
        fx = torch.fft.fft(x, dim=-1)
        phase_spec = torch.angle(fx)
        phase_shift = torch.pi / 2
        modified_phase_spec = phase_spec + phase_shift
        modified_freq = torch.abs(fx) * torch.exp(1j * modified_phase_spec)
        x = torch.fft.ifft(modified_freq, dim=-1).real
        # Weight the shifted embeddings
        x = x * weight
        return x

"""### Causal Self-Attention"""

## Self Attention Block - Karpathy
class CausalSelfAttention(nn.Module):
  def __init__(self, config):
      super().__init__()
      assert config.n_embd % config.n_head == 0
      # key, query, value projections for all heads, but in a batch
      self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
      # output projection
      self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
      # regularization
      self.attn_dropout = nn.Dropout(config.dropout)
      self.resid_dropout = nn.Dropout(config.dropout)
      self.n_head = config.n_head
      self.n_embd = config.n_embd
      self.dropout = config.dropout
      # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
      self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
      if not self.flash:
          print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
          # causal mask to ensure that attention is only applied to the left in the input sequence
          self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                      .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # This line is retrieving the dimensions of the input tensor x. B is the batch size (the number of sequences in the batch), T is the sequence length (the number of elements in each sequence), and C is the embedding dimension (the size of the vector representation of each element).
    # This line is first passing the input x through a linear transformation defined by self.c_attn, then splitting the result into three equal-sized parts along the last dimension (dim=2). These three parts correspond to the query (q), key (k), and value (v) tensors used in the attention mechanism.
    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    # This line is reshaping and transposing the key tensor k to prepare it for the multi-head attention mechanism. self.n_head is the number of attention heads. The view() function changes the shape of the tensor to (B, T, self.n_head, C // self.n_head), meaning it divides the last dimension (the embedding dimension) into self.n_head parts. The transpose(1, 2) function swaps the second and third dimensions, so that the attention head dimension comes before the sequence length dimension.
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    # This line does the same thing for the query tensor q as the previous line does for the key tensor k.
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    # This line does the same thing for the value tensor v as the previous line does for the query tensor q.
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    if self.flash:
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.resid_dropout(self.c_proj(y))
    return y

"""### Gaussian Self-Attention"""

class GaussianSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    # regularization
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)
    self.config = config
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.mu = nn.Parameter(torch.zeros(1))
    self.sigma = nn.Parameter(torch.ones(1))

    # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
        print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    # batch size, sequence length, embedding dimensions
    B, T, C = x.size()

    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    if self.flash:
      # efficient attention using Flash Attention CUDA kernels
      y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    else:
      # manual implementation of attention
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
      att = F.softmax(att, dim=-1)
      att = self.attn_dropout(att)
      y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1,2).contiguous().view(B, T, C)
    # print(f'Y Shape after dotP transpose: {y.shape}')
    # T-GSA: Compute Gaussian weights
    gauss_weights = torch.exp(-(torch.arange(T, device=x.device).float().unsqueeze(0) - self.mu[:, None])**2 / (2 * self.sigma[:, None]**2))
    gauss_weights = gauss_weights / gauss_weights.sum(dim=-1, keepdim=True)  # normalize weights
    y = y * gauss_weights.unsqueeze(-1)
    # print(f'Y Shape after Gauss Unsqueeze: {y.shape}')
    # output projection
    y = self.resid_dropout(self.c_proj(y))
    # print(f'Y Out Shape: {y.shape}')
    return y

"""### Convolutional - 1D"""

# Looking at modifying the attention function to include guassian weights
class Convolve(nn.Module):
  def __init__(self, config):
      super().__init__()
      self.config = config
      self.conv1d_r = nn.Conv1d(self.config.n_embd, self.config.n_embd, kernel_size=3, padding=1, groups=self.config.n_embd)
      self.conv1d_i = nn.Conv1d(self.config.n_embd, self.config.n_embd, kernel_size=3, padding=1, groups=self.config.n_embd)


  def forward(self, x):
    xr = x.real
    xi = x.imag

    # Change the shape to (batch_size, num_channels, length)
    xr = xr.permute(0, 2, 1)
    xi = xi.permute(0, 2, 1)

    # Apply the 1D convolution.
    xr = self.conv1d_r(xr)
    xi = self.conv1d_i(xi)

    # Apply an activation function - I like sigmoids like swish here.


    # Change the shape back to (batch_size, length, num_channels)
    xr = xr.permute(0, 2, 1)
    xi = xi.permute(0, 2, 1)

    x = torch.complex(xr.float(), xi.float())

    return x.real.bfloat16()

"""### Fourier Block"""

class FourierBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.norm = nn.LayerNorm(self.config.n_embd)


    self.ff = nn.Sequential(
        nn.Linear(self.config.n_embd, self.config.n_embd, bias=self.config.bias),
        nn.GELU(),
        nn.Linear(self.config.n_embd, self.config.n_embd, bias=self.config.bias),
        nn.Dropout(self.config.dropout),
    )

  def forward(self, x):
    print(f'==== FourierBlock ==== x.size @ entry: {x.size()}')
    xf = x.float()
    print(f"==== FourierBlock ==== to Float xf: {xf.size()}")
    # FFT on the positional embeddings looking for patterns and periodicities in the token position sequence
    fx = torch.fft.fft(xf, dim=-1).real
    #print(f'==== FourierBlock ==== fx(fft) fx.size: {fx.size()}')
    x = self.norm(fx)
    print(f'==== FourierBlock ==== x(norm) x.size: {x.size()}')
    fx = self.ff(x)
    print(f'==== FourierBlock ==== fx(feed forward) fx.size: {fx.size()}')
    x = self.norm(fx)
    print(f'==== FourierBlock ==== x(norm) x.size outgoing: {x.size()}')
    return x

"""### fNet Training Block"""

# The forward pass of this block is modeled after Google's implementation from their FNet paper with the addition of Karpathy's Causal Self-Attention layer.
class fNetBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.norm = nn.LayerNorm(self.config.n_embd)
    self.four = FourierBlock(self.config)
    self.p_attent = PhasedAttention(self.config)

    self.ff = nn.Sequential(
        nn.Linear(self.config.n_embd, 1 * self.config.n_embd, bias=self.config.bias),
        nn.GELU(),
        nn.Linear(1 * self.config.n_embd, self.config.n_embd, bias=self.config.bias),
        nn.Dropout(self.config.dropout),
    )

  def forward(self, x):
    print(f'==== fNetBlock ==== x.size @ entry: {x.size()}')
    x = self.four(x)
    print(f'==== fNetBlock ==== FourierBlock Return x.size: {x.size()}')
    # x = self.p_attent(x)
    x = self.ff(x)
    print(f'==== fNetBlock ==== x.size Outgoing: {x.size()}')
    return x


  def printBlock(self, x, function):
    print(f'----------------')
    print(f'{function}: {x.size()}')
    print(f'----------------')
    print(f'x: {x}')

"""### fNet"""

@dataclass
class fNetConfig:
    block_size: int = 1024
    batch_size: int = 2
    n_layer: int = 1
    n_head: int = 1
    dropout: float = 0
    n_embd: int = 1536
    vocab_size: int = 100000
    bias: bool = False

def get_completion():
  start = "Please tell me more about "
  num_samples = 1
  max_new_tokens = 256
  temperature = 0.75
  top_k = 200
  enc = tiktoken.get_encoding("cl100k_base")
  encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
  decode = lambda l: enc.decode(l)

  start_ids = encode(start)
  x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

  with torch.no_grad():
    with ctx:
      for k in range(num_samples):
        y= fNet().generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        sample = decode(y[0].tolist())
        print(sample)

  return sample

class fNet(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.config = config
      self.layers = nn.ModuleList([fNetBlock(self.config) for _ in range(self.config.n_layer)])

      self.transformer = nn.ModuleDict(dict(
          drop = nn.Dropout(self.config.dropout),
          wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
          wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
          norm = LayerNorm(self.config.n_embd, bias=self.config.bias),
      ))

      self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
      self.transformer.wte.weight = self.lm_head.weight

      # init all weights
      self.apply(self._init_weights)

      for pn, p in self.named_parameters():
          if pn.endswith('c_proj.weight'):
              torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

      # report number of parameters
      print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding=True):
      n_params = sum(p.numel() for p in self.parameters())

      if non_embedding:
          n_params -= self.transformer.wpe.weight.numel()
      return n_params

    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          module.weight.data.normal_(mean=0.0, std=0.02)
          if isinstance(module, nn.Linear) and module.bias is not None:
              module.bias.data.zero_()

    def forward(self, x, targets=None):
      device = x.device
      print(f"==== fNet Fwd ==== Device: {device}")
      b, t = x.size() # batch_size, block_size
      # print(f"fNet forward b: {b} t: {t}")
      # print(f"fNet Forward X Device: {device}")
      assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
      pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
      print(f'===== fNet Forward ==== x.size @ block input: {x.size()} ==== !! =====')

      tok_emb = self.transformer.wte(x)  # (vocab_size, n_embd)  there are 50304 different token embeddings (which corresponds to the size of the vocabulary) and each embedding is also 768-dimensional.
      print(f'==== fNet Forward ==== Token Embedding Size: {tok_emb.size()}')
      pos_emb = self.transformer.wpe(x)  # (block_size, n_embd) there are 1024 different position embeddings (which is the maximum sequence length the model can handle) and each embedding is 768-dimensional.
      print(f'==== fNet Forward ==== Position Embedding Size: {pos_emb.size()}')
      x = self.transformer.drop(tok_emb + pos_emb)  # simple addition of the two embeddings element-wise. tok_emb and pos_emb have the same shape, and the result of the addition will also have the same shape, (batch_size, sequence_length, 768). This essentially combines the token and position information for each token in the block/sequence.
      print(f'==== fNet Forward ==== Token + Position Embedding Size: {x.size()}')

      for block in self.layers:
        x = block(x)
        print(f'==== fNet Forward Layer Block RETURNED! ==== x output: {x.size()}')

      x = self.transformer.norm(x)
      print(f'==== fNet Forward ==== x norm: {x.size()}')

      if targets is not None:
          logits = self.lm_head(x)
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
      else:
          logits = self.lm_head(x[:, [-1], :])
          loss = None

      return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
      # start with all of the candidate parameters
      param_dict = {pn: p for pn, p in self.named_parameters()}

      # filter out those that do not require grad
      param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

      # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
      # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
      decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
      nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

      optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},
          {'params': nodecay_params, 'weight_decay': 0.0}
      ]

      num_decay_params = sum(p.numel() for p in decay_params)
      num_nodecay_params = sum(p.numel() for p in nodecay_params)
      print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
      print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

      # Create AdamW optimizer and use the fused version if it is available
      fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
      use_fused = fused_available and device_type == 'cuda'
      extra_args = dict(fused=True) if use_fused else dict()
      optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
      print(f"using fused AdamW: {use_fused}")

      return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
      """
      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.
      """
      for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
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

"""## Model Config & Env Setup"""

# 'new' or 'resume'
init_from = 'new'

# Num simultaneous
batch_size = 1

# Context length
block_size = 1024

# Number of layers in the model
n_layer = 1

# Number of heads - Causal Self-Attention
n_head = 1

# Embedding dim
n_embd = 1536

# Dropout Rate
dropout = 0.00

# Bias : boolean
bias = False

# Weights Directory
out_dir = '/content/drive/MyDrive/Datasets/openwebtext/out'

# Checkpointing and eval
eval_interval = 250
log_interval = 1
eval_iters = 25
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 * 1 # used to simulate larger batch sizes

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 100000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(f"config: {config_keys}")
# -----------------------------------------------------------------------------

master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1342)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
print(device_type)

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = '/content/drive/MyDrive/Datasets/openwebtext'
train_data = np.memmap('/content/drive/MyDrive/Datasets/openwebtext/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('/content/drive/MyDrive/Datasets/openwebtext/val.bin', dtype=np.uint16, mode='r')
out_dir = '/content/drive/MyDrive/Datasets/openwebtext/out'


def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
  if device_type == 'cuda':
      # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
      x, y = x.pin_memory().to(device, non_blocking=False), y.pin_memory().to(device, non_blocking=False)
  else:
      x, y = x.to(device), y.to(device)

  # print(f"Batch X Device: {x.device}, Batch Y Device: {y.device}")
  return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(block_size=block_size, vocab_size=None, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  dropout=dropout, bias=bias, batch_size=batch_size) # start with model_args from command line

if init_from == 'new':
  # init a new model from scratch
  print("Initializing a new model")
  # determine the vocab size we'll use for from-scratch training
  print("vocab_size of cl100k_base to around 100k (100000 rounded for efficiency)")
  model_args['vocab_size'] = 100000
  fconfig = fNetConfig(**model_args)
  model = fNet(fconfig)

elif init_from == 'resume':
  print(f"Resuming training from {out_dir}")
  # resume training from a checkpoint.
  ckpt_path = f'{out_dir}/ckpt.pt'
  checkpoint = torch.load(ckpt_path, map_location=device)
  checkpoint_model_args = checkpoint['model_args']
  # print(f"Checkpoint Model Args: {checkpoint_model_args}")
  # force these config attributes to be equal otherwise we can't even resume training
  # the rest of the attributes (e.g. dropout) can stay as desired from command line
  for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']:
      model_args[k] = checkpoint_model_args[k]
  # create the model
  fconfig = fNetConfig(**model_args)
  model = fNet(fconfig)
  state_dict = checkpoint['model']
  # print(f"Checkpoint Model State Dict: {state_dict}")
  # fix the keys of the state dictionary :(
  # honestly no idea how checkpoints sometimes get this prefix, have to debug more
  unwanted_prefix = '_orig_mod.'

  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

  model.load_state_dict(state_dict)
  print(state_dict.keys())
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']

model.to(device)
# print(f"Next Model Params Device: {next(model.parameters()).device}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
  print("compiling the model... (takes a ~minute)")
  unoptimized_model = model
  model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      # print(f"Est Loss X Device: {X.device}, Est Loss Y Device: {Y.device}")
      print(f'==== Estimate Loss ==== k range eval  X: {X.size()} Y: {Y.size()}')
      with ctx:
        logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
  # 1) linear warmup for warmup_iters steps
  if it < warmup_iters:
      return learning_rate * it / warmup_iters
  # 2) if it > lr_decay_iters, return min learning rate
  if it > lr_decay_iters:
      return min_lr
  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # fetch the very first batch
# print(f"Training 1 X Device: {X.device}, Training 1 Y Device: {Y.device}")
t0 = time.time()

local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0

"""## Training Loop"""

while True:
  # determine and set the learning rate for this iteration
  lr = get_lr(iter_num) if decay_lr else learning_rate
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr

  # evaluate the loss on train/val sets and write checkpoints
  if iter_num % eval_interval == 0 and master_process:
      losses = estimate_loss()
      print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
      if losses['val'] < best_val_loss or always_save_checkpoint:
          best_val_loss = losses['val']
          if iter_num > 0:
              checkpoint = {
                  'model': raw_model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'model_args': model_args,
                  'iter_num': iter_num,
                  'best_val_loss': best_val_loss,
                  'config': config,
              }
              print(f"saving checkpoint to {out_dir}")
              torch.save(checkpoint, f'{out_dir}/ckpt.pt')
  if iter_num == 0 and eval_only:
      break

  # forward backward update, with optional gradient accumulation to simulate larger batch size
  # and using the GradScaler if data type is float16
  for micro_step in range(gradient_accumulation_steps):
    ddp = False
    if ddp:
          # in DDP training we only need to sync gradients at the last micro step.
          # the official way to do this is with model.no_sync() context manager, but
          # I really dislike that this bloats the code and forces us to repeat code
          # looking at the source of that context manager, it just toggles this variable
          model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
  # clip the gradient
  if grad_clip != 0.0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
  # step the optimizer and scaler if training in fp16
  scaler.step(optimizer)
  scaler.update()
  # flush the gradients as soon as we can, no need for this memory anymore
  optimizer.zero_grad(set_to_none=True)

  # timing and logging
  t1 = time.time()
  dt = t1 - t0
  t0 = t1
  if iter_num % log_interval == 0 and master_process:
    # get loss as float. note: this is a CPU-GPU sync point
    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    lossf = loss.item() * gradient_accumulation_steps

    if local_iter_num >= 5:  # let the training loop settle a bit
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

  iter_num += 1
  local_iter_num += 1

  # termination conditions
  if iter_num > max_iters:
      break

"""## Generate"""

sample = True

if sample:
  get_completion()