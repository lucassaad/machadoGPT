import os 
import pickle
import argparse

import numpy as np
import torch 
import torch.nn as nn
from torch.nn import functional as F

from prepare import DataPreparer

INPUT_DIR = os.path.join(os.path.dirname(__file__), "inputs-bin")
OUT_DIR = 'out'


parser = argparse.ArgumentParser()
parser.add_argument('init_from', choices=['scratch', 'resume'])
parser.add_argument('tipo_literario', choices=['texto', 'poesia', 'teatro'])
args = parser.parse_args()

init_from = args.init_from
tipo_literario = args.tipo_literario


# hyperparameters

batch_size = 12         # how many independent sequences will we process in parallel?
block_size = 64         # what is the maximum context length for predictions?
max_iters = 40000         # total number of iterarions
eval_interval = 500     # every 500 iterations the model is valueated
learning_rate = 3e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# quando chega a hora de avaliar (a cada eval_interval), 
# quantos batches são usados para calcular a média da loss
eval_iters = 20
# o tamanho do vetor que representa cada token internamente no modelo
n_embd = 128
# número de cabeças de atenção
n_head = 8
# quantos blocos transformer (attention + feedforward) são empilhados. 
# Mais camadas = modelo mais profundo, capaz de aprender padrões mais complexos,
n_layer = 8
# probabilidade (20%) de "desligar" aleatoriamente neurônios durante o treino
# como técnica de regularização para evitar overfitting
dropout = 0.0

torch.manual_seed(1337)


def load_dataset(tipo_literario: str):
    input_path = os.path.join(INPUT_DIR, tipo_literario)

    # carrega metadados
    with open(os.path.join(input_path, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta["vocab_size"]
    stoi = meta["stoi"]
    itos = meta["itos"]

    # carrega train e val respeitando o dtype usado na escrita (uint16)
    train_arr = np.fromfile(os.path.join(input_path, "train.bin"), dtype=np.uint16)
    val_arr = np.fromfile(os.path.join(input_path, "val.bin"), dtype=np.uint16)

    train_data = torch.tensor(train_arr, dtype=torch.long)
    val_data = torch.tensor(val_arr, dtype=torch.long)

    return vocab_size, stoi, itos, train_data, val_data


input_path = os.path.join(INPUT_DIR, tipo_literario)
if not os.path.exists(os.path.join(input_path, "meta.pkl")):
    DataPreparer().prepare(tipo_literario)

vocab_size, stoi, itos, train_data, val_data = load_dataset(tipo_literario)


def get_batch(split: str):
    data = train_data if split == 'train' else val_data

    # {batch_size} inteiros aleatorios entre [0 : len(data) - block_size]
    ix = torch.randint(len(data) - block_size, (batch_size,))

    inputs = torch.stack([data[i:i+block_size] for i in ix])
    targets = torch.stack([data[i+1:i+1+block_size] for i in ix])
    # exemplo:
    # inputs:  [...0,4,5,9...]
    # targets: [...4,5,9,3...]

    inputs, targets = inputs.to(device), targets.to(device)

    return inputs, targets
            

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
            nn.ReLU(),  # non-linear
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


if __name__ == "__main__":
    
    out_dir = os.path.join(OUT_DIR, tipo_literario)
    checkpoint_dir = os.path.join('checkpoints', tipo_literario)
    best_val_loss = 1e9
    checkpoint = None
    iter_num = 0

    model = GPTLanguageModel()

    if init_from == 'resume':
        print(f"Resuming training from {checkpoint_dir}")
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        print("Initializing a new model from scratch")

    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer']) # TypeIgnore
        checkpoint = None
    
    for iter in range(iter_num, iter_num + max_iters + 1):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter,
                    'best_val_loss': best_val_loss,
                }, os.path.join(checkpoint_dir, 'ckpt.pt'))
                print(f"checkpoint salvo em {checkpoint_dir}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    decode = lambda l: [itos[i] for i in l]
    out = ''.join(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/ouput.txt', 'w') as f:
        f.write(out)
    
