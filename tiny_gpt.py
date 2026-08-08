"""
Tiny SQL GPT — a ~1M parameter transformer, built from random weights,
trained on a laptop, that writes SQL you can actually run.

No pretrained weights. No HuggingFace model classes. Just PyTorch tensors.

Read this file top to bottom and you have seen the whole path:
    §1 schema      what the SQL is about
    §2 data        we GENERATE the training set (nothing scraped)
    §3 tokenizer   text -> integers
    §4 model       embeddings -> causal attention -> MLP -> logits
    §5 bigram      the dumb baseline that makes the GPT number mean something
    §6 train       next-token prediction + backprop
    §7 generate    sampling, temperature, top-k
    §8 explain     print the internals: mask, softmax, attention
    §9 cli

    python tiny_gpt.py --data          # build the dataset
    python tiny_gpt.py --train         # train the ~1M param model (~3 min CPU)
    python tiny_gpt.py --generate 10   # write some SQL
    python tiny_gpt.py --explain       # open the black box
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
SEED = 1337

# Set True to keep attention matrices around for the interpretability probe (§8).
# Off during training because [B, H, T, T] per layer is a lot of wasted memory.
SAVE_ATTN = False


# ─────────────────────────────────────────────────────────────────────────────
# §1  SCHEMA — three small tables. Deliberately generic so anyone can read it.
# ─────────────────────────────────────────────────────────────────────────────

# Four categorical columns per table is deliberate. With only one groupable
# column per table the model could learn a shortcut ("orders -> GROUP BY status")
# instead of the actual rule ("copy the SELECT column"). Four columns makes the
# shortcut useless, so the held-out test below measures real generalization.
SCHEMA = {
    "sales":     {"cat": ["region", "product", "segment", "quarter"],
                  "num": ["qty", "price", "day"]},
    "customers": {"cat": ["city", "tier", "source", "plan"],
                  "num": ["age", "spend"]},
    "orders":    {"cat": ["status", "channel", "priority", "carrier"],
                  "num": ["total", "items"]},
}

def _q(*words):
    return [f"'{w}'" for w in words]


VALUES = {
    "region":  _q("north", "south", "east", "west",
                  "central", "coastal", "inland", "northeast"),
    "product": _q("widget", "gadget", "gizmo", "doohickey",
                  "sprocket", "cog", "lever", "valve"),
    "city":    _q("austin", "denver", "boston", "seattle",
                  "chicago", "portland", "atlanta", "phoenix"),
    "tier":    _q("gold", "silver", "bronze", "platinum",
                  "basic", "premium", "trial", "legacy"),
    "status":  _q("open", "shipped", "closed", "pending",
                  "cancelled", "returned", "draft", "backorder"),
    "channel": _q("web", "store", "phone", "partner",
                  "kiosk", "mobile", "email", "reseller"),
    "segment": _q("retail", "wholesale", "online", "direct",
                  "enterprise", "smb", "government", "education"),
    "quarter": _q("q1", "q2", "q3", "q4", "h1", "h2", "fy", "ytd"),
    "source":  _q("ads", "referral", "organic", "outbound",
                  "social", "events", "affiliate", "search"),
    "plan":    _q("monthly", "annual", "free", "team",
                  "business", "starter", "pro", "custom"),
    "priority": _q("low", "medium", "high", "urgent",
                   "critical", "routine", "deferred", "escalated"),
    "carrier": _q("ups", "fedex", "dhl", "usps",
                  "freight", "local", "courier", "air"),
}

AGGS = ["SUM", "AVG", "MAX", "MIN"]
THRESHOLDS = ["10", "50", "100", "200", "250", "500", "750", "1000", "1500", "2000"]
LIMITS = ["1", "3", "5", "10", "20", "25", "50", "100"]

# THE GENERALIZATION TEST.
# These (table, column) pairs NEVER appear in a GROUP BY during training.
# The columns themselves do appear elsewhere (SELECT, WHERE), so they have
# embeddings — the model just never saw them grouped.
#
# At eval we prompt "SELECT channel , COUNT ( * ) FROM orders GROUP BY" and ask:
# does it say `channel`? If yes, it learned the RULE, not the pairs.
HELD_OUT_GROUPBY = [("orders", "channel"), ("sales", "product"), ("customers", "tier")]


# ─────────────────────────────────────────────────────────────────────────────
# §2  DATA — we generate every training example from a grammar we control.
#
# Why generated and not scraped:
#   - no licensing questions
#   - a learner can read the ENTIRE source of the training data (it's right here)
#   - we can inject the long-range dependency on purpose (GROUP BY agreement)
#   - reproducible from a seed
#   - we know the exact training set, so we can MEASURE memorization
# ─────────────────────────────────────────────────────────────────────────────

def _table_cols(t):
    return SCHEMA[t]["cat"] + SCHEMA[t]["num"]


def make_query(rng, allow_heldout=False):
    """Emit one space-separated SQL query. Shape picked at random."""
    t = rng.choice(list(SCHEMA))
    cats, nums = SCHEMA[t]["cat"], SCHEMA[t]["num"]
    cat, cat2 = rng.choice(cats), rng.choice(cats)
    num, num2 = rng.choice(nums), rng.choice(nums)
    agg = rng.choice(AGGS)
    val = rng.choice(VALUES[cat2])
    lim = rng.choice(LIMITS)
    thr, thr2 = rng.choice(THRESHOLDS), rng.choice(THRESHOLDS)

    # For GROUP BY shapes, pick a grouping column that is NOT a held-out
    # (table, column) pair. This is what creates the unseen test cases.
    choices = [c for c in cats if allow_heldout or (t, c) not in HELD_OUT_GROUPBY]
    g = rng.choice(choices) if choices else None

    shape = rng.randint(0, 13)

    if shape == 0:
        return f"SELECT {cat} FROM {t} ;"
    if shape == 1:
        return f"SELECT * FROM {t} LIMIT {lim} ;"
    if shape == 2:
        return f"SELECT {cat} FROM {t} WHERE {cat2} = {val} ;"
    if shape == 3:
        return f"SELECT {num} FROM {t} WHERE {num} > {thr} ;"
    if shape == 4:
        return f"SELECT COUNT ( * ) FROM {t} WHERE {cat2} = {val} ;"
    if shape == 5:
        return f"SELECT {cat} FROM {t} ORDER BY {num} DESC LIMIT {lim} ;"
    if shape == 6:
        return f"SELECT {agg} ( {num} ) FROM {t} ;"
    if shape == 7:
        return f"SELECT {cat} , {num} FROM {t} WHERE {cat2} = {val} ;"
    if shape == 8:
        return (f"SELECT {cat} FROM {t} WHERE {cat2} = {val} "
                f"AND {num} > {thr} ;")
    if shape == 9:
        return f"SELECT {agg} ( {num} ) FROM {t} WHERE {num2} > {thr} ;"

    if g is None:                       # every column of this table is held out
        return f"SELECT {cat} FROM {t} ;"

    if shape == 10:
        return f"SELECT {g} , COUNT ( * ) FROM {t} GROUP BY {g} ;"
    if shape == 11:
        return f"SELECT {g} , {agg} ( {num} ) FROM {t} GROUP BY {g} ;"
    if shape == 12:
        return (f"SELECT {g} , {agg} ( {num} ) FROM {t} "
                f"WHERE {cat2} = {val} GROUP BY {g} ;")
    # The big one: WHERE + AND + GROUP BY + ORDER BY + LIMIT. Long enough that
    # keeping GROUP BY agreeing with SELECT is a genuine long-range dependency.
    return (f"SELECT {g} , {agg} ( {num} ) FROM {t} "
            f"WHERE {cat2} = {val} AND {num2} > {thr2} "
            f"GROUP BY {g} ORDER BY {agg} ( {num} ) DESC LIMIT {lim} ;")


def build_dataset(n=100_000, seed=SEED, out=None):
    rng = random.Random(seed)
    queries = [make_query(rng) for _ in range(n)]
    os.makedirs(DATA_DIR, exist_ok=True)
    out = out or os.path.join(DATA_DIR, "queries.txt")
    with open(out, "w") as f:
        f.write("\n".join(queries))
    manifest = {
        "seed": seed,
        "n_queries": n,
        "unique_queries": len(set(queries)),
        "held_out_groupby": [list(p) for p in HELD_OUT_GROUPBY],
        "schema": SCHEMA,
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return queries, manifest


def load_queries():
    path = os.path.join(DATA_DIR, "queries.txt")
    if not os.path.exists(path):
        raise SystemExit("No dataset. Run: python tiny_gpt.py --data")
    with open(path) as f:
        return f.read().splitlines()


# ─────────────────────────────────────────────────────────────────────────────
# §3  TOKENIZER — text becomes integers. That is the whole job.
#
# Word-level, because SQL is already emitted space-separated. ~110 tokens total,
# which is the point: a vocabulary this small means we can print the ENTIRE
# probability distribution at every step (§8). No frontier model can do that.
# ─────────────────────────────────────────────────────────────────────────────

BOS = "<s>"


class Tokenizer:
    def __init__(self, queries):
        vocab = {BOS}
        for q in queries:
            vocab.update(q.split())
        self.itos = sorted(vocab)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[t] for t in text.split()]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)


def build_corpus(queries, tok):
    """One long stream of token ids: <s> q1 ; <s> q2 ; ... — trained on windows."""
    ids = []
    bos = tok.stoi[BOS]
    for q in queries:
        ids.append(bos)
        ids.extend(tok.encode(q))
    return torch.tensor(ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# §4  MODEL — a decoder-only transformer. This is the part people pretend
#     to understand. It is about 90 lines.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    name: str = "tiny"
    vocab_size: int = 0
    block_size: int = 64      # context window, in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


# The four rungs of the scaling ladder (§4b of PLAN.md). Same data, same code.
SIZES = {
    "nano":  dict(n_layer=1, n_head=2, n_embd=32),
    "micro": dict(n_layer=2, n_head=4, n_embd=64),
    "tiny":  dict(n_layer=4, n_head=4, n_embd=128),
    "small": dict(n_layer=6, n_head=8, n_embd=256),
}


class CausalSelfAttention(nn.Module):
    """Every token looks back at earlier tokens and decides what matters.

    The causal mask is what makes this a *language* model: position t may
    attend to 0..t, never to the future. Without it the model would cheat by
    reading the answer it is being asked to predict.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # q, k, v in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # lower-triangular ones: mask[i][j] == 1 means "i may look at j"
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.att_cache = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if SAVE_ATTN:
            self.att_cache = att.detach()
        att = self.drop(att)

        y = att @ v                                        # [B,H,T,head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # merge heads back
        return self.drop(self.proj(y))


class Block(nn.Module):
    """LayerNorm -> attention -> residual, then LayerNorm -> MLP -> residual.

    The residual (+x) is why deep networks train at all: gradients get a
    clean path back to the input.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # what the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # where it sits
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"context is {self.cfg.block_size}, got {T}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)      # [B, T, n_embd]
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))            # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Predict token t+1 from tokens 0..t, at every position at once.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, stop=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]   # context window: hard limit
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if stop is not None and (nxt == stop).all():
                break
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# §5  BIGRAM BASELINE — "what token usually follows this one", no attention.
#     Its job is to be bad. A number is only meaningful next to a baseline.
# ─────────────────────────────────────────────────────────────────────────────

class Bigram:
    def __init__(self, vocab_size):
        self.counts = torch.ones(vocab_size, vocab_size)   # +1 smoothing

    def fit(self, ids):
        for a, b in zip(ids[:-1].tolist(), ids[1:].tolist()):
            self.counts[a, b] += 1
        return self

    def generate(self, start, max_new_tokens, stop=None):
        out = [start]
        for _ in range(max_new_tokens):
            probs = self.counts[out[-1]] / self.counts[out[-1]].sum()
            nxt = int(torch.multinomial(probs, 1))
            out.append(nxt)
            if nxt == stop:
                break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# §6  TRAIN — sample random windows, predict the next token, backpropagate.
# ─────────────────────────────────────────────────────────────────────────────

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])  # shifted by one
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, batch_size, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(cfg, splits, steps=3000, batch_size=64, lr=1e-3, device="cpu",
          log_every=500, quiet=False):
    torch.manual_seed(SEED)
    model = TinyGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []

    if not quiet:
        print(f"[{cfg.name}] {model.n_params():,} params "
              f"(L{cfg.n_layer} H{cfg.n_head} E{cfg.n_embd}) on {device}")

    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            losses = estimate_loss(model, splits, cfg, batch_size, device)
            history.append({"step": step, **losses})
            if not quiet:
                print(f"  step {step:5d}  train {losses['train']:.4f}  "
                      f"val {losses['val']:.4f}")
        x, y = get_batch(splits["train"], cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    return model, history


def make_splits(corpus, frac=0.9):
    n = int(frac * len(corpus))
    return {"train": corpus[:n], "val": corpus[n:]}


def save_ckpt(model, tok, history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "cfg": asdict(model.cfg),
        "state_dict": model.state_dict(),
        "itos": tok.itos,
        "history": history,
    }, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tok = Tokenizer.__new__(Tokenizer)
    tok.itos = ck["itos"]
    tok.stoi = {s: i for i, s in enumerate(tok.itos)}
    return model, tok, ck.get("history", [])


# ─────────────────────────────────────────────────────────────────────────────
# §7  GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def sample_queries(model, tok, n=10, temperature=0.8, top_k=None, device="cpu"):
    bos, semi = tok.stoi[BOS], tok.stoi[";"]
    out = []
    for _ in range(n):
        idx = torch.tensor([[bos]], dtype=torch.long, device=device)
        idx = model.generate(idx, max_new_tokens=model.cfg.block_size - 1,
                             temperature=temperature, top_k=top_k, stop=semi)
        out.append(tok.decode(idx[0, 1:].tolist()))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §8  EXPLAIN — open the black box. This is the teaching payload.
# ─────────────────────────────────────────────────────────────────────────────

def bar(p, width=28):
    return "█" * int(round(p * width))


def explain(model, tok, device="cpu"):
    prompt = "SELECT region , SUM ( qty ) FROM sales GROUP BY"
    ids = [tok.stoi[BOS]] + tok.encode(prompt)

    print("\n" + "=" * 68)
    print("1. VOCABULARY — the model's entire universe")
    print("=" * 68)
    print(f"vocab size : {len(tok)} tokens")
    print(f"sample     : {', '.join(tok.itos[:14])} ...")
    if model:
        print(f"parameters : {model.n_params():,}")
        print(f"context    : {model.cfg.block_size} tokens  "
              f"(L{model.cfg.n_layer} H{model.cfg.n_head} E{model.cfg.n_embd})")

    print("\n" + "=" * 68)
    print("2. TOKENS — text is not words to a model, it is integers")
    print("=" * 68)
    print(f"text   : {prompt}")
    print(f"tokens : {ids[1:]}")
    print(f"count  : {len(ids)} tokens (including <s>)")

    print("\n" + "=" * 68)
    print("3. CAUSAL MASK — why it cannot see the future")
    print("=" * 68)
    k = min(8, len(ids))
    names = [tok.itos[i] for i in ids[:k]]
    w = max(len(s) for s in names) + 1
    print(" " * w + "".join(f"{s[:5]:>6}" for s in names))
    for i, s in enumerate(names):
        row = "".join(f"{'  1' if j <= i else '  .':>6}" for j in range(k))
        print(f"{s:>{w}}" + row)
    print("\n  1 = may attend   . = masked out (the future)")

    if model is None:
        print("\n(no checkpoint — run --train for sections 4 and 5)\n")
        return

    x = torch.tensor([ids], dtype=torch.long, device=device)
    global SAVE_ATTN
    SAVE_ATTN = True
    with torch.no_grad():
        logits, _ = model(x)
    SAVE_ATTN = False

    print("\n" + "=" * 68)
    print("4. THE FULL DISTRIBUTION — the model outputs probabilities, not answers")
    print("=" * 68)
    print("Same model, same softmax, two positions. Confidence is not a")
    print("property of the model — it is a property of the context.\n")

    for label, ctx in [
        ("CONSTRAINED — only one column can legally follow", prompt),
        ("OPEN — any table column could come next", "SELECT"),
    ]:
        cids = torch.tensor([[tok.stoi[BOS]] + tok.encode(ctx)],
                            dtype=torch.long, device=device)
        with torch.no_grad():
            cl, _ = model(cids)
        row = cl[0, -1]
        probs = F.softmax(row, dim=-1)
        print(f"  context: ...{ctx[-46:]}")
        print(f"  {label}")
        top = torch.topk(probs, 6)
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"    {tok.itos[i]:<12} {p:6.3f}  {bar(p)}")
        print(f"    {'(other ' + str(len(tok) - 6) + ')':<12} "
              f"{1 - top.values.sum().item():6.3f}")
        print("    temperature reshapes this distribution — nothing else:")
        for t in (0.2, 1.0, 2.0):
            pt = F.softmax(row / t, dim=-1)
            tt = torch.topk(pt, 3)
            line = "  ".join(f"{tok.itos[i]}:{p:.2f}"
                             for p, i in zip(tt.values.tolist(), tt.indices.tolist()))
            print(f"      T={t:<4} {line}")
        print()

    print("\n" + "=" * 68)
    print("5. ATTENTION — what the last token actually looked at")
    print("=" * 68)
    print(f"query: {prompt}")
    print("position of the final token: predicting the GROUP BY column\n")
    for li, blk in enumerate(model.blocks):
        att = blk.attn.att_cache
        if att is None:
            continue
        for h in range(att.shape[1]):
            row = att[0, h, -1, :]
            top = torch.topk(row, 3)
            parts = "  ".join(
                f"{tok.itos[ids[i]]}({p:.2f})"
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            )
            print(f"  L{li}H{h}  ->  {parts}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# §9  CLI
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description="Tiny SQL GPT")
    ap.add_argument("--data", action="store_true", help="generate the dataset")
    ap.add_argument("--n", type=int, default=100_000, help="dataset size")
    ap.add_argument("--train", action="store_true", help="train a model")
    ap.add_argument("--size", default="tiny", choices=list(SIZES))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--generate", type=int, metavar="N", help="sample N queries")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--explain", action="store_true", help="open the black box")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ckpt = os.path.join(CKPT_DIR, f"{args.size}.pt")

    if args.data:
        queries, man = build_dataset(args.n)
        print(f"wrote {man['n_queries']:,} queries "
              f"({man['unique_queries']:,} unique) to data/queries.txt")
        print(f"held out from GROUP BY: {HELD_OUT_GROUPBY}")
        print("\nsamples:")
        for q in queries[:5]:
            print(f"  {q}")
        return

    if args.train:
        queries = load_queries()
        tok = Tokenizer(queries)
        corpus = build_corpus(queries, tok)
        print(f"{len(queries):,} queries  {len(corpus):,} tokens  "
              f"vocab {len(tok)}")
        cfg = Config(name=args.size, vocab_size=len(tok), **SIZES[args.size])
        model, hist = train(cfg, make_splits(corpus), steps=args.steps,
                            device=device)
        save_ckpt(model, tok, hist, ckpt)
        print(f"saved {ckpt}")
        print("\nsamples:")
        for q in sample_queries(model, tok, 5, device=device):
            print(f"  {q}")
        return

    if not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint at {ckpt}. Run: python tiny_gpt.py --train")
    model, tok, _ = load_ckpt(ckpt, device)

    if args.generate:
        for q in sample_queries(model, tok, args.generate,
                                temperature=args.temperature, device=device):
            print(q)
        return

    if args.explain:
        explain(model, tok, device)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
