# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

print('Hyperparameters Updated')

torch.manual_seed(1337)

with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
print('Characters: ',chars)
vocab_size = len(chars)
print('Vocabulary Size: ',vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print('Iteration: ',iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model_2.pth')
'''# Recreate the model architecture
loaded_model = GPTLanguageModel()
loaded_model.load_state_dict(torch.load('model_1.pth'))
loaded_model.to(device)
loaded_model.eval()
print("Model loaded from gpt_music_model.pth")'''

context = torch.zeros((1, 1), dtype=torch.long, device=device)
gpt_melody = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(gpt_melody)

import math

def compute_perplexity(melody, model):
    tokenized = [stoi[ch] for ch in melody if ch in stoi]

    if len(tokenized) < block_size:
        raise ValueError("Melody length must be greater than or equal to block_size.")

    x_batches = []
    y_batches = []
    for i in range(0, len(tokenized) - block_size):
        x = tokenized[i:i + block_size]
        y = tokenized[i + 1:i + 1 + block_size]
        x_batches.append(torch.tensor(x, dtype=torch.long))
        y_batches.append(torch.tensor(y, dtype=torch.long))

    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for x, y in zip(x_batches, y_batches):
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return perplexity

import random

def generate_random_melody(length, vocab):
    return ''.join(random.choices(vocab, k=length))

random_melody = generate_random_melody(500, chars)
print("Random Melodies:")
print(random_melody)

from collections import Counter
def compute_token_frequencies(data, chars):
    counts = Counter(data)
    total = sum(counts.values())
    probabilities = {char: counts[char] / total for char in chars}
    return probabilities

freq_probs = compute_token_frequencies(list(text), chars)
print("Frequencies:", freq_probs)
def generate_freq_based_melody(length, freq_probs):
    vocab, probs = zip(*freq_probs.items())
    return ''.join(random.choices(vocab, probs, k=length))

freq_melody = generate_freq_based_melody(500, freq_probs)
print("\nFrequency-Based Melodies:")
print(freq_melody)

from collections import defaultdict

def build_ngram_model(data, n):
    ngrams = defaultdict(Counter)
    for i in range(len(data) - n):
        context = tuple(data[i:i+n-1])
        next_token = data[i+n-1]
        ngrams[context][next_token] += 1

    ngram_probs = {context: {token: count / sum(counter.values())
                             for token, count in counter.items()}
                   for context, counter in ngrams.items()}
    return ngram_probs

ngram_model = build_ngram_model(list(text), n=3)

def generate_ngram_melody(length, ngram_model, vocab, n):
    melody = [random.choice(vocab) for _ in range(n-1)]
    for _ in range(length - (n-1)):
        context = tuple(melody[-(n-1):])
        next_token = random.choices(
            list(ngram_model[context].keys()),
            weights=list(ngram_model[context].values()),
            k=1
        )[0] if context in ngram_model else random.choice(vocab)
        melody.append(next_token)
    return ''.join(melody)

ngram_melody = generate_ngram_melody(500, ngram_model, chars, n=3)
print("N-Gram Melody:", ngram_melody)

gpt_perplexity = compute_perplexity(gpt_melody, model)
random_perplexity = compute_perplexity(random_melody, model)
freq_perplexity = compute_perplexity(freq_melody, model)
ngram_perplexity = compute_perplexity(ngram_melody, model)

print(f"GPT Model Perplexity: {gpt_perplexity}")
print(f"Random Baseline Model Perplexity: {random_perplexity}")
print(f"Frequency Based Baseline Model Perplexity: {freq_perplexity}")
print(f"N-Gram Baseline Model Perplexity: {ngram_perplexity}")

sample_val_melody = decode(val_data[:500].tolist())
print(sample_val_melody)

gpt_val_perplexity = compute_perplexity(sample_val_melody, model)
print(f"GPT Model Perplexity on Validation Data: {gpt_val_perplexity}")

from nltk.metrics.distance import edit_distance

def compute_edit_distance(seq1, seq2):
    return edit_distance(seq1, seq2)

def compute_token_overlap(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def compute_sequence_similarity(seq1, seq2, weight=0.5):
    edit_dist = compute_edit_distance(seq1, seq2)
    token_overlap = compute_token_overlap(seq1, seq2)
    normalized_edit_distance = 1 - (edit_dist / max(len(seq1), len(seq2)))
    return weight * token_overlap + (1 - weight) * normalized_edit_distance

models = {
    "GPT": gpt_melody,
    "Random": random_melody,
    "Frequency": freq_melody,
    "N-gram": ngram_melody
}

results = {}

for model_name, melody in models.items():
    edit_dist = compute_edit_distance(melody, sample_val_melody)
    token_overlap = compute_token_overlap(melody, sample_val_melody)
    seq_similarity = compute_sequence_similarity(melody, sample_val_melody)
    results[model_name] = {
        "Edit Distance": edit_dist,
        "Token Overlap": token_overlap,
        "Sequence Similarity": seq_similarity
    }

# Display Results
for model, metrics in results.items():
    print(f"Model: {model}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

pip install simpleaudio

from pydub import AudioSegment
import numpy as np
import simpleaudio as sa


NOTE_FREQUENCIES = {
    'C': 261.63,
    'c': 277.18,  # C#
    'D': 293.66,
    'd': 311.13,  # D#
    'E': 329.63,
    'F': 349.23,
    'f': 369.99,  # F#
    'G': 392.00,
    'g': 415.30,  # G#
    'A': 440.00,
    'a': 466.16,  # A#
    'B': 493.88,
    'R': 0     # Rest
}


# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    wave = 0.5 * amplitude * np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(),
        frame_rate=sample_rate,
        sample_width=wave.dtype.itemsize,
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes
def create_sequence(note_sequence, duration_ms=500):
    song = AudioSegment.silent(duration=0)
    for note in note_sequence:
        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        else:
            frequency = NOTE_FREQUENCIES[note]
            segment = generate_sine_wave(frequency, duration_ms)
        song += segment
    return song

song = create_sequence(gpt_melody.replace('\n', ''), duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("gpt_melody.wav", format="wav")

# Play the .wav file using simpleaudio
'''wave_obj = sa.WaveObject.from_wave_file("gpt_melody.wav")
play_obj = wave_obj.play()
play_obj.wait_done()'''

song = create_sequence(random_melody.replace('\n', ''), duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("random_melody.wav", format="wav")

# Play the .wav file using simpleaudio
'''wave_obj = sa.WaveObject.from_wave_file("random_melody.wav")
play_obj = wave_obj.play()
play_obj.wait_done()'''

song = create_sequence(freq_melody.replace('\n', ''), duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("freq_melody.wav", format="wav")

# Play the .wav file using simpleaudio
'''wave_obj = sa.WaveObject.from_wave_file("freq_melody.wav")
play_obj = wave_obj.play()
play_obj.wait_done()'''

song = create_sequence(ngram_melody.replace('\n', ''), duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("ngram_melody.wav", format="wav")

# Play the .wav file using simpleaudio
'''wave_obj = sa.WaveObject.from_wave_file("ngram_melody.wav")
play_obj = wave_obj.play()
play_obj.wait_done()'''

from itertools import product

def grid_search(param_grid):

  best_params = {}
  best_val_loss = float('inf')

  for params in product(*param_grid.values()):

    global learning_rate, batch_size, dropout, n_layer, n_head
    learning_rate = params[0]
    batch_size = params[1]
    dropout = params[2]
    n_layer = params[3]
    n_head = params[4]
    n_embd = params[5]


    print('learning_rate=',learning_rate)
    print('batch_size=',batch_size)
    print('dropout=',dropout)
    print('n_layer=',n_layer)
    print('n_head=',n_head)
    print('n_embd=',n_embd)

    model2 = GPTLanguageModel2()
    m2 = model2.to(device)

    print(sum(p.numel() for p in m2.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model2.parameters(), lr=learning_rate)

    val_loss = estimate_loss2()['val']
    print(val_loss)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_params = dict(zip(param_grid.keys(), params))

  return best_params, best_val_loss

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [3e-4, 1e-3, 5e-4],
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'n_layer': [4, 6, 8],
    'n_head': [4, 8, 16],
    'n_embd' : [512, 768]  }


best_params, best_val_loss = grid_search(param_grid)
print(f"Best Hyperparameters: {best_params}")
print(f"Best Validation Loss: {best_val_loss}")

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 500
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 8
dropout = 0.3
# ------------

print('Hyperparameters Updated')

with open('inputMelodiesAugmentedv2.txt', 'r', encoding='utf-8') as f2:
    text2 = f2.read()

# here are all the unique characters that occur in this text
chars2 = sorted(list(set(text2)))
print('Characters: ',chars2)
vocab_size2 = len(chars2)
print('Vocabulary Size: ',vocab_size2)
# create a mapping from characters to integers
stoi2 = { ch:i for i,ch in enumerate(chars2) }
itos2 = { i:ch for i,ch in enumerate(chars2) }
encode2 = lambda s: [stoi2[c] for c in s] # encoder: take a string, output a list of integers
decode2 = lambda l: ''.join([itos2[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data2 = torch.tensor(encode2(text2), dtype=torch.long)
n = int(0.9*len(data2)) # first 90% will be train, rest val
train_data2 = data2[:n]
val_data2 = data2[n:]

# data loading
def get_batch2(split):
    # generate a small batch of data of inputs x and targets y
    data2 = train_data2 if split == 'train' else val_data2
    ix = torch.randint(len(data2) - block_size, (batch_size,))
    x = torch.stack([data2[i:i+block_size] for i in ix])
    y = torch.stack([data2[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss2():
    out = {}
    model2.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch2(split)
            logits, loss = model2(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model2.train()
    return out



class Head2(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.relative_positions = nn.Embedding(2 * block_size - 1, head_size)

        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.randn(block_size, head_size))
        #self.tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)


         # Compute relative positional embeddings
        pos_emb = self.pos_emb[:T, :].unsqueeze(0)  # (1, T, head_size)
        relative_wei = torch.einsum('bth,tlh->btl', q, pos_emb)  # (B, T, T)

        # Standard self-attention with relative positional encoding
        wei = (q @ k.transpose(-2, -1) + relative_wei) * (C ** -0.5)  # (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply causal masking
        wei = wei.masked_fill(self.tril[:T, :T].to(x.device) == 0, float('-inf'))  # Apply causal masking



        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)


        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention2(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head2(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward2(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block2(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention2(n_head, head_size)
        self.ffwd = FeedFoward2(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel2(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.pitch_embedding_table = nn.Embedding(vocab_size2, n_embd // 2)
        self.time_embedding_table = nn.Embedding(block_size, n_embd // 2)
        self.combined_projection = nn.Linear(n_embd, n_embd)

        self.blocks = nn.Sequential(*[Block2(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size2)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pitch_emb = self.pitch_embedding_table(idx)  # (B, T, n_embd//2)
        time_emb = self.time_embedding_table(torch.arange(T, device=device))  # (T, n_embd//2)

        # Combine embeddings
        combined_emb = torch.cat((pitch_emb, time_emb.unsqueeze(0).expand(B, -1, -1)), dim=-1)
        x = self.combined_projection(combined_emb)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model2 = GPTLanguageModel2()
m2 = model2.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m2.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model2.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print('Iteration: ',iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss2()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch2('train')

    # evaluate the loss
    logits, loss = model2(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model2.state_dict(), 'model_5.pth')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
gpt_melody2 = decode2(m2.generate(context, max_new_tokens=500)[0].tolist())
print(gpt_melody2)

def compute_perplexity2(melody, model):
    tokenized = [stoi2[ch] for ch in melody if ch in stoi2]

    if len(tokenized) < block_size:
        raise ValueError("Melody length must be greater than or equal to block_size.")

    x_batches = []
    y_batches = []
    for i in range(0, len(tokenized) - block_size):
        x = tokenized[i:i + block_size]
        y = tokenized[i + 1:i + 1 + block_size]
        x_batches.append(torch.tensor(x, dtype=torch.long))
        y_batches.append(torch.tensor(y, dtype=torch.long))

    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for x, y in zip(x_batches, y_batches):
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)  # Add batch dim
            _, loss = model(x, y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return perplexity

sample_val_melody2 = decode2(val_data2[500:1000].tolist())
print(sample_val_melody2)

gpt_val_perplexity_2 = compute_perplexity(sample_val_melody2, model2)
print(f"GPT Model Perplexity on Validation Data: {gpt_val_perplexity_2}")
gpt_perplexity_2 = compute_perplexity(gpt_melody2, model2)
print(f"GPT Model Perplexity on Validation Data: {gpt_perplexity_2}")

models = {
    "GPT": gpt_melody2
}

results = {}

for model_name, melody in models.items():
    edit_dist = compute_edit_distance(melody, sample_val_melody2)
    token_overlap = compute_token_overlap(melody, sample_val_melody2)
    seq_similarity = compute_sequence_similarity(melody, sample_val_melody2)
    results[model_name] = {
        "Edit Distance": edit_dist,
        "Token Overlap": token_overlap,
        "Sequence Similarity": seq_similarity
    }

# Display Results
for model, metrics in results.items():
    print(f"Model: {model}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

song = create_sequence(gpt_melody2.replace('\n', ''), duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("gpt_melody_updated.wav", format="wav")

