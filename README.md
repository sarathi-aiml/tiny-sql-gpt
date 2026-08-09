# Tiny SQL GPT

> **The goal:** build a language model from scratch, end to end, small enough to understand every
> part of it, and in doing so understand how the large ones actually work.

**A 841,216-parameter transformer, built from random weights, trained on a laptop CPU in 5 minutes.
100% of the SQL it generates actually executes.**

No pretrained weights. No HuggingFace model classes. No API keys. ~700 lines of Python you can read
in one sitting.

> **New to what's inside a language model?** Start with **[EXPLAIN.md](EXPLAIN.md)**, the same
> project with no maths and no jargon, in five minutes. This page is the detailed version.

```
$ python tiny_gpt.py --generate 3
SELECT MAX ( items ) FROM orders ;
SELECT quarter FROM sales WHERE product = 'gizmo' ;
SELECT city , MIN ( age ) FROM customers WHERE plan = 'annual' GROUP BY city ;
```

---

## How it works, in five steps

Each step is one section of `tiny_gpt.py` and one command. Nothing happens off-screen.

```
 1  BUILD THE CORPUS                            tiny_gpt.py §2
    python tiny_gpt.py --data

    A hand-written grammar emits 100,000 SQL statements across 14 query
    shapes. Synthetic and seeded, so the corpus is reproducible and every
    pattern in it is known. Three (table, column) pairs are withheld from the
    GROUP BY position as a held-out generalization probe.
    out:  16,941 unique statements, 1.3M tokens, 90/10 train/val split


 2  TOKENIZE: TEXT -> TOKEN IDS -> VECTORS      tiny_gpt.py §3

    Word-level tokenizer. Keywords, identifiers, literals and punctuation each
    map to one integer index in a 155-entry vocabulary.

      "SELECT region , SUM"   ->   [147, 99, 131, 96]

    Those integers are not features, they are row indices. The token embedding
    matrix W_e (155 x 128) is looked up per ID to produce a dense vector in
    R^128, and a learned positional embedding W_p (64 x 128) is added so the
    model encodes both identity and position:

      x = W_e[ids] + W_p[0..T)            x : [B, T, 128]


 3  THE MODEL: DECODER-ONLY TRANSFORMER         tiny_gpt.py §4

    d_model 128 | 4 heads | head_dim 32 | 4 blocks | context 64 | pre-LN
    Written directly in PyTorch. No transformers library, no AutoModel,
    no pretrained weights.

      x : [B, T, 128]
        └─► Block x4
              ├─ LayerNorm
              │    └─ causal self-attention
              │         q,k,v = x @ W_qkv          q,k,v : [B, 4, T, 32]
              │         A     = softmax( q k^T / sqrt(32) + mask )
              │         out   = (A @ v) @ W_o          A : [B, 4, T, T]
              │    └─ residual add
              └─ LayerNorm
                   └─ MLP  128 -> 512 -> 128, GELU
                   └─ residual add
        └─► final LayerNorm
        └─► unembedding W_u (128 x 155)     logits : [B, T, 155]

    The mask is strictly lower-triangular, so position t attends only to 0..t.
    That is what makes this autoregressive rather than bidirectional.
    out:  841,216 parameters, fp32


 4  TRAIN: NEXT-TOKEN PREDICTION                tiny_gpt.py §6
    python tiny_gpt.py --train

    Objective is cross-entropy between the logits at position t and the token
    at t+1, evaluated at every position in parallel (teacher forcing):

      loss = cross_entropy( logits.view(-1, 155), targets.view(-1) )

    AdamW (lr 1e-3, weight decay 0.01), cosine decay, grad-norm clip 1.0,
    batches of 64 sequences x 64 tokens sampled as random windows.
    out:  loss 4.62 -> 0.662 val, ~5 min on a laptop CPU, no GPU


 5  DECODE AND EVALUATE BY EXECUTION            inference.py / evaluate.py
    python evaluate.py --eval

    Autoregressive decoding: forward pass, take logits[:, -1], scale by
    temperature, optional top-k truncation, softmax to a distribution, sample,
    append, repeat until the ';' token. No KV cache; at 64 tokens of context
    the recompute is cheaper than the code to avoid it.

    Scoring is execution-based, not string similarity. Every generated
    statement is run against a live SQLite database and graded on whether it
    executes, against a bigram baseline for reference.
    out:  the numbers below
```

Steps 1 to 4 all live in one file, `tiny_gpt.py`, in that order. Read it top to bottom and you
follow a SQL statement through tokenization, embedding, attention, backprop, and back out as
generated text.

---

## Results

Every number below is produced by `python evaluate.py --eval`, which generates 500 queries and
runs them against a real SQLite database. Nothing is graded by eye, and the run is seeded, so you
will get these exact numbers.

| metric | TinyGPT (841K params) | bigram baseline |
|---|---:|---:|
| parses | **100.0%** | 4.4% |
| **executes** | **100.0%** | 4.4% |
| `GROUP BY` agrees with `SELECT` | **100.0%** | 3.4% |
| novel (not in training set) | 16.6% | 98.6% |
| validation loss | 0.662 | n/a |

The bigram baseline is there on purpose. A number without a baseline is decoration.

100% is a real measurement, not a rounding flourish, but read it against the task. The grammar has
14 query shapes over a 3-table schema and a 155-token vocabulary. A model that saturates *this* is
not a text-to-SQL system; it's proof that the training loop works. The interesting number is the
16.6% novelty, and the failure in the next section.

---

## Where the model actually is

Fair question, and the one this repo gets asked most. **The model is one file: `tiny_gpt.py`.**
Every other Python file in the repo is scaffolding around it. None of them contain a model.

| file | contains the model? | what it's for |
|---|---|---|
| **`tiny_gpt.py`** | **yes, all of it** | schema, data generation, tokenizer, model, training loop, sampling, `--explain` |
| `evaluate.py` | no | runs generated SQL against SQLite, scaling curve, attention probe |
| `inference.py` | no | loads a checkpoint and generates. Imports the model from `tiny_gpt.py` |
| `test_tiny_gpt.py` | no | 12 tests |
| `push_to_hub.py` | no | packages and uploads to Hugging Face |

Inside `tiny_gpt.py`, the model itself is §4, roughly 120 lines covering `CausalSelfAttention`,
`Block`, and `TinyGPT`. Everything imported is `torch`, `torch.nn`, and the standard library.
There is no `transformers`, no `AutoModel`, no pretrained anything.

### How it was actually trained

Exactly these commands, in this order, on a MacBook CPU, with no GPU and no cloud:

```bash
python tiny_gpt.py --data                        # 100,000 queries      ~2 sec
python tiny_gpt.py --train --size tiny --steps 3000   # 841,216 params  ~5 min
python evaluate.py --eval                        # 500 queries vs SQLite  ~1 min
python evaluate.py --attention                   # probe all 16 heads    instant
python evaluate.py --scaling --steps 3000        # all four sizes       ~40 min
```

Training is `torch.optim.AdamW` over random 64-token windows of the corpus, cross-entropy on the
next token, 3,000 steps at batch 64. Loss went `4.62 → 0.66`. That is the entire training story.
The loop is ~25 lines in §6 and you can read all of it.

---

## Three things this repo does that a tutorial doesn't

### 1. An executable metric

Generated SQL is run against a real database. It works or it doesn't. No human judgement, no
vibes. This is the difference between "look, it makes plausible text" and a measurement.

### 2. A scaling curve, and it does not say what you'd expect

The same architecture at four sizes, same data, same code, all trained on one laptop.

| model | params | val loss | executes | **`GROUP BY` agrees** |
|---|---:|---:|---:|---:|
| nano | 24,736 | 0.697 | 99.2% | **36.6%** |
| micro | 124,032 | 0.668 | 99.8% | **100.0%** |
| tiny | 841,216 | 0.662 | 100.0% | 100.0% |
| small | 4,834,816 | 0.661 | 100.0% | 100.0% |

![scaling](figures/scaling.png)

Three things fall out of this, and none of them is "bigger is better":

**Syntax is nearly free.** `nano`, at 24,736 parameters and one layer, writes SQL that executes 99.2%
of the time. Surface fluency is the cheapest thing a language model learns.

**The long-range dependency is what costs parameters.** That same `nano` gets `GROUP BY` agreement
right 36.6% of the time, which is essentially the ~33% you'd get by picking a groupable column at
random.
It learned what SQL *looks like* and almost nothing about the rule. Between 24K and 124K parameters
it goes to 100%. That is a phase transition you can watch happen on a laptop.

**Then it flattens.** Going from `micro` to `small` is 39x more parameters for 0.007 of validation
loss and no change in either behavioural metric. All four converge to ~0.66 because ~0.66 is the
entropy of the data generator: it picks tables, columns and values at random, and no model can
predict a coin flip. That floor is a property of the data, not a limitation of the models.

The useful question was never "is bigger better." It's **where does the curve bend for my task**,
because past that point, more parameters buy nothing. That's measurable in an afternoon.

#### An ablation, and a hypothesis that died

`nano` → `micro` changed depth *and* width at once, so the jump above is confounded. My hypothesis
was depth: copying a token from earlier in the sequence sounds like it needs one attention operation
to locate it and a second to move it.

So I trained `flat`: **one layer**, widened to match `micro`'s parameter count:

| model | layers | params | `GROUP BY` agrees |
|---|---:|---:|---:|
| nano | 1 | 24,736 | 36.6% |
| **flat** | **1** | **127,160** | **99.5%** |
| micro | 2 | 124,032 | 100.0% |

**99.5% against 100.0%.** At matched parameters, depth bought essentially nothing. The hypothesis
was wrong, and the nano→micro jump was capacity all along. One layer is enough.

In hindsight it's clear why: attending from "just after `GROUP BY`" to "just after `SELECT`" can be
done from position and syntax alone. There's no previous-token head to compose with. Copying
arbitrary *novel* bigrams is the harder job, and that's the case that needs two layers.

Reproduce with `python tiny_gpt.py --train --size flat` and `python evaluate.py --eval --size flat`.

### 3. An interpretability finding, and a caveat worth more than the finding

The training data has a deliberate long-range dependency: the column after `GROUP BY` is always
the column that appeared first in `SELECT`. Learning that requires looking back ~8 tokens.

`python evaluate.py --attention` scores every head on how much attention it pays to that column.
Run it on the smallest model that can actually do the job, `flat`, one layer and 127K parameters:

```
query: SELECT region , SUM ( qty ) FROM sales GROUP BY

  L0H1   0.587  ███████████████████████
  ...
  best: layer 0, head 1, 58.7% of its attention lands on the SELECT column
  uniform baseline would be 8.3%  (7.0x)
```

**One head, doing one job.** Nobody designed or labelled it. The bright cell on the `BY` row is the
head reaching back to `region`, and the blank upper triangle is the causal mask:

![attention](figures/attention-flat.png)

Now run the same probe on `tiny`, 4 layers and 841K parameters, which gets `GROUP BY` agreement
right **100%** of the time:

```
  L1H0   0.177  ███████
  L3H2   0.122  █████
  L1H1   0.071  ███
  L0H1   0.070  ███
  ...
  best: layer 1, head 0, 17.7% of its attention
  uniform baseline would be 8.3%  (2.1x)

  VERDICT: no single head owns this dependency, it is distributed.
```

**Same behaviour. No legible circuit.** The bigger model is not worse at the task, it is worse at
being read. The one-layer model has nowhere else to put the computation, so it is forced into a
single head. The four-layer model smears it across heads and layers, and the tidy picture
disappears.

That is the real lesson, and it is inconvenient:

> Interpretability is not a property of the behaviour. It is a property of the model that happens
> to implement it. Scale does not just add capability, it dissolves the clean circuits you were
> hoping to read.

**A caveat on top of the caveat.** This is seed-sensitive. An earlier training run of `tiny` put a
single head above 85% on this exact probe. The current seeded run does not reproduce that at all.
Treat any single-head circuit claim in a small model with suspicion unless it replicates across
seeds, including the ones in this README.

---

## The honest negative

Three `(table, column)` pairs were **never** grouped during training: `orders.channel`,
`sales.product`, `customers.tier`. The columns appear everywhere else, just never after `GROUP BY`.
So: did the model learn the *rule*, or the *pairs*?

```
  [MISS] SELECT channel ... FROM orders    GROUP BY -> status
         p(channel)=0.0014  rank 4/155  |  control p=0.000368  ->  copy lift   3.8x
  [MISS] SELECT product ... FROM sales     GROUP BY -> segment
         p(product)=0.0026  rank 4/155  |  control p=0.000524  ->  copy lift   5.0x
  [MISS] SELECT tier    ... FROM customers GROUP BY -> source
         p(tier)=0.0041    rank 4/155  |  control p=0.000399  ->  copy lift  10.2x

  0/3 correct on unseen pairs, mean copy lift 6.3x
```

**0 out of 3.** But look closer before calling it a failure.

*Copy lift* compares `p(col | SELECT col ... GROUP BY)` against a control prompt with a different
`SELECT` column. Naming the column in `SELECT` raises its `GROUP BY` probability by **3.8x to
10.2x**, so the context *is* being used. It just loses to a blanket prior against tokens that
never appeared in that slot during training.

That is the whole story of hallucination, at 841K parameters:

> **Attention identifies the right source token. The output prior overrules it.**

The model is 100% correct on columns it has seen grouped, and confidently wrong on ones it hasn't.
It never learned a rule. It learned a very good lookup table. Scaling this up doesn't change the
mechanism; it just makes the lookup table bigger.

---

## Quickstart

```bash
pip install -r requirements.txt

python tiny_gpt.py --data              # generate 100,000 SQL queries (~2s)
python tiny_gpt.py --train             # train 841K params (~5 min, CPU)
python tiny_gpt.py --explain           # open the black box

python inference.py                    # write some SQL
python inference.py --run              # write it AND execute it against SQLite
python inference.py --probs "SELECT region , SUM ( qty ) FROM sales GROUP BY"

python space/app.py                    # local Gradio demo in your browser

python evaluate.py --eval              # the headline number
python evaluate.py --attention         # probe all 16 heads
python evaluate.py --scaling           # the full curve + chart
python evaluate.py --plot              # re-render the chart from saved results

python test_tiny_gpt.py                # 12 tests, no framework
```

Trained checkpoints for `nano`, `micro`, `flat` and `tiny` are committed, so `--generate`,
`--explain`, `--eval`, `--attention` and the ablation all work without training anything.
`small` (19 MB) is not committed. `--scaling` reuses the checkpoints it finds and trains only
`small`, about 30 minutes on a laptop CPU. Everything else in `--scaling` is instant.

Sizes: `--size nano | micro | flat | tiny | small`.

---

## Open the black box

`python tiny_gpt.py --explain` prints the internals on real data: vocabulary, token IDs, the
causal mask, the **complete** probability distribution, and per-head attention.

The vocabulary is 155 tokens, which is the point: small enough to print the *entire* softmax.
No frontier model demo can do this.

```
3. CAUSAL MASK: why it cannot see the future

          <s> SELEC regio     ,   SUM     (   qty     )
    <s>     1     .     .     .     .     .     .     .
 SELECT     1     1     .     .     .     .     .     .
 region     1     1     1     .     .     .     .     .
      ,     1     1     1     1     .     .     .     .
    SUM     1     1     1     1     1     .     .     .
      (     1     1     1     1     1     1     .     .
    qty     1     1     1     1     1     1     1     .
      )     1     1     1     1     1     1     1     1

4. THE FULL DISTRIBUTION: the model outputs probabilities, not answers

Same model, same softmax, two positions. Confidence is not a property of the
model, it is a property of the context.

  context: ...SELECT region , SUM ( qty ) FROM sales GROUP BY
  CONSTRAINED: only one column can legally follow
    region        0.996  ████████████████████████████
    segment       0.001
    product       0.001
    (other 149)   0.001

  context: ...SELECT
  OPEN: any table column could come next
    *             0.072  ██
    COUNT         0.070  ██
    region        0.064  ██
    segment       0.064  ██
    (other 149)   0.603
```

Same model. 0.996 versus 0.072.

---

## How it works

```
 ┌─ DATA ─────────────────────────────────────────────────────────────┐
 │  A grammar we wrote (tiny_gpt.py §2) emits 100,000 SQL queries.    │
 │  Nothing scraped. Reproducible from seed 1337. 16,941 unique.      │
 │  The GROUP BY dependency is injected on purpose.                   │
 └────────────────────────────────┬───────────────────────────────────┘
                                  ▼
    "SELECT region , SUM ( qty ) FROM sales GROUP BY region ;"
                                  │  split on whitespace  →  155-token vocab
                                  ▼
              [130, 147, 99, 131, 96, 145, 97, 124, 148, 125, 121]
                                  │
                                  ▼
    tok_emb [155,128]  +  pos_emb [64,128]   ──►  x [B, T, 128]
                                  │
      ┌───────────────────────────┴────────────────────────────┐
      │  BLOCK × 4                                             │
      │    x ──► LayerNorm ──► Causal Self-Attention ──►(+)──┐ │
      │    └───────────────── residual ──────────────────────┘ │
      │    x ──► LayerNorm ──► MLP(128→512→128) ────────►(+)──┐ │
      │    └───────────────── residual ──────────────────────┘ │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
                  LayerNorm ──► lm_head [128 → 155]
                                  ▼
                        logits [B, T, 155]
                                  ▼
                  cross_entropy(logits, next_token)  ──►  loss 0.662
```

Inside one attention head:

```
  x ──► Linear(128→384) ──► split ──► q, k, v   each [B, 4, T, 32]
                                        │
                    att = q @ kᵀ / √32  │  →  [B, 4, T, T]
                                        ▼
              CAUSAL MASK: position t may see 0..t, never t+1
                                        ▼
                     softmax ──► @ v ──► merge heads ──► Linear
```

---

## Layout

```
EXPLAIN.md         the no-jargon version. start here if you build with LLMs
                   but have never looked inside one.

tiny_gpt.py        §1 schema  §2 data  §3 tokenizer  §4 model
                   §5 bigram  §6 train §7 generate   §8 explain  §9 cli
inference.py       run the model. no training code, no eval harness.
evaluate.py        executable eval · scaling curve · attention probe
test_tiny_gpt.py   12 tests, plain asserts

space/app.py       Gradio demo: generate, execute, full softmax, all 16 heads
push_to_hub.py     package + publish to the Hugging Face Hub
hf/MODEL_CARD.md   the model card template

PLAN.md            the design doc this was built from
POSTS.md           write-ups drafted from these results
data/              generated corpus + manifest (seed 1337)
checkpoints/       trained models
figures/           scaling curve, attention heatmaps
```

The core model lives in **one file** on purpose. The teaching value dies the moment a reader has to
jump across eight modules to follow one token through the network.

---

## Tests

```
$ python test_tiny_gpt.py
  ok  test_causal_mask_blocks_the_future        # change a future token, earlier logits must not move
  ok  test_checkpoint_round_trips
  ok  test_dataset_is_reproducible
  ok  test_generation_produces_parseable_output
  ok  test_groupby_rule_detector
  ok  test_heldout_pairs_never_grouped_in_training   # the held-out pairs really are held out
  ok  test_log_every_does_not_change_the_trained_model  # eval must not consume the training RNG
  ok  test_param_count_matches_hand_calc        # 841,216 against a hand-derived formula
  ok  test_shapes_match_the_diagram
  ok  test_sqlite_harness_grades_correctly
  ok  test_tokenizer_round_trips
  ok  test_training_reduces_loss

12 passed
```

---

## What this is not

- Not a text-to-SQL product. It generates SQL; it does not answer questions.
- Not a general language model. 155 tokens of vocabulary, one toy schema.
- Not a scaling-laws paper. Four points on a curve is a demonstration, not a finding.
- Not novel research. Every component here is standard. The point is that it is *legible*.

## What it is

A complete, honest, end-to-end path from a dataset you can read to a token you can explain,
with a metric, a baseline, and a negative result that wasn't cherry-picked away.

