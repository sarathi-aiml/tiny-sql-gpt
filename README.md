# Tiny SQL GPT

**A 841,216-parameter transformer, built from random weights, trained on a laptop CPU in 5 minutes.
99.8% of the SQL it generates actually executes.**

No pretrained weights. No HuggingFace model classes. No API keys. ~700 lines of Python you can read
in one sitting.

```
$ python tiny_gpt.py --generate 3
SELECT segment , COUNT ( * ) FROM sales GROUP BY segment ;
SELECT plan , MIN ( age ) FROM customers WHERE tier = 'platinum' GROUP BY plan ;
SELECT status , SUM ( total ) FROM orders WHERE status = 'backorder' GROUP BY status ;
```

---

## Results

Every number below is produced by `python evaluate.py --eval`, which generates 500 queries and
runs them against a real SQLite database. Nothing is graded by eye.

| metric | TinyGPT (841K params) | bigram baseline |
|---|---:|---:|
| parses | **99.8%** | 2.8% |
| **executes** | **99.8%** | 2.8% |
| `GROUP BY` agrees with `SELECT` | **100.0%** | 4.8% |
| novel (not in training set) | 11.8% | 99.0% |
| validation loss | 0.661 | — |

The bigram baseline is there on purpose. A number without a baseline is decoration.

---

## Three things this repo does that a tutorial doesn't

### 1. An executable metric

Generated SQL is run against a real database. It works or it doesn't — no human judgement, no
vibes. This is the difference between "look, it makes plausible text" and a measurement.

### 2. A scaling curve

The same architecture at four sizes, same data, same code, all trained on one laptop.

<!-- SCALING_TABLE -->

![scaling](figures/scaling.png)

### 3. An interpretability finding

The training data has a deliberate long-range dependency: the column after `GROUP BY` is always
the column that appeared first in `SELECT`. Learning that requires looking back ~8 tokens.

`python evaluate.py --attention` probes all 16 heads:

```
query: SELECT region , SUM ( qty ) FROM sales GROUP BY
       does any head look back at 'region' when predicting the next token?

  L1H1   0.925  █████████████████████████████████████
  L1H3   0.899  ████████████████████████████████████
  L1H0   0.534  █████████████████████
  L2H1   0.154  ██████
  ...
  L3H1   0.000

  best: layer 1, head 1 — 92.5% of its attention lands on the SELECT column
  uniform baseline would be 8.3%  (11.1x)
```

**Layer 1, head 1 learned `GROUP BY` agreement.** Not a textbook diagram — an actual head in an
actual model, found on a laptop, reproducible with one command.

![attention](figures/attention.png)

---

## The honest negative

Three `(table, column)` pairs were **never** grouped during training — `orders.channel`,
`sales.product`, `customers.tier`. The columns appear everywhere else, just never after `GROUP BY`.
So: did the model learn the *rule*, or the *pairs*?

```
  [MISS] SELECT channel ... FROM orders GROUP BY -> carrier
         p(channel)=0.0018  rank 6/155  |  control p=0.000462  ->  copy lift  3.9x
  [MISS] SELECT product ... FROM sales    GROUP BY -> segment
         p(product)=0.0006  rank 12/155  |  control p=0.000362  ->  copy lift  1.7x
  [MISS] SELECT tier    ... FROM customers GROUP BY -> plan
         p(tier)=0.0021    rank 4/155   |  control p=0.000262  ->  copy lift  7.9x

  0/3 correct on unseen pairs — mean copy lift 4.5x
```

**0 out of 3.** But look closer before calling it a failure.

*Copy lift* compares `p(col | SELECT col ... GROUP BY)` against a control prompt with a different
`SELECT` column. Naming the column in `SELECT` raises its `GROUP BY` probability by **4.5x** — so
the copy circuit found by L1H1 *is* firing. It just loses to a blanket prior against tokens that
never appeared in that slot during training.

That is the whole story of hallucination, at 841K parameters:

> **Attention identifies the right source token. The output prior overrules it.**

The model is 100% correct on columns it has seen grouped, and confidently wrong on ones it hasn't.
It never learned a rule — it learned a very good lookup table. Scaling this up doesn't change the
mechanism; it just makes the lookup table bigger.

---

## Quickstart

```bash
pip install -r requirements.txt

python tiny_gpt.py --data              # generate 100,000 SQL queries (~2s)
python tiny_gpt.py --train             # train 841K params (~5 min, CPU)
python tiny_gpt.py --generate 10       # write some SQL
python tiny_gpt.py --explain           # open the black box

python evaluate.py --eval              # the headline number
python evaluate.py --attention         # probe all 16 heads
python evaluate.py --scaling           # train all 4 sizes + chart (~40 min)

python test_tiny_gpt.py                # 11 tests, no framework
```

A trained checkpoint is committed, so `--generate`, `--explain`, `--eval` and `--attention` all
work without training anything.

---

## Open the black box

`python tiny_gpt.py --explain` prints the internals on real data — vocabulary, token IDs, the
causal mask, the **complete** probability distribution, and per-head attention.

The vocabulary is 155 tokens, which is the point: small enough to print the *entire* softmax.
No frontier model demo can do this.

```
3. CAUSAL MASK — why it cannot see the future

          <s> SELEC regio     ,   SUM     (   qty     )
    <s>     1     .     .     .     .     .     .     .
 SELECT     1     1     .     .     .     .     .     .
 region     1     1     1     .     .     .     .     .
      ,     1     1     1     1     .     .     .     .
    SUM     1     1     1     1     1     .     .     .
      (     1     1     1     1     1     1     .     .
    qty     1     1     1     1     1     1     1     .
      )     1     1     1     1     1     1     1     1

4. THE FULL DISTRIBUTION — the model outputs probabilities, not answers

Same model, same softmax, two positions. Confidence is not a property of the
model — it is a property of the context.

  context: ...SELECT region , SUM ( qty ) FROM sales GROUP BY
  CONSTRAINED — only one column can legally follow
    region        0.995  ████████████████████████████
    segment       0.002
    quarter       0.001
    (other 149)   0.001

  context: ...SELECT
  OPEN — any table column could come next
    *             0.070  ██
    COUNT         0.069  ██
    status        0.064  ██
    plan          0.064  ██
    (other 149)   0.607
```

Same model. 0.995 versus 0.070.

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
                  cross_entropy(logits, next_token)  ──►  loss 0.661
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
tiny_gpt.py        §1 schema  §2 data  §3 tokenizer  §4 model
                   §5 bigram  §6 train §7 generate   §8 explain  §9 cli
evaluate.py        executable eval · scaling curve · attention probe
test_tiny_gpt.py   11 tests, plain asserts
PLAN.md            the design doc this was built from
data/              generated corpus + manifest (seed 1337)
checkpoints/       trained models
figures/           scaling curve, attention heatmap
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
  ok  test_param_count_matches_hand_calc        # 841,216 against a hand-derived formula
  ok  test_shapes_match_the_diagram
  ok  test_sqlite_harness_grades_correctly
  ok  test_tokenizer_round_trips
  ok  test_training_reduces_loss

11 passed
```

---

## What this is not

- Not a text-to-SQL product. It generates SQL; it does not answer questions.
- Not a general language model. 155 tokens of vocabulary, one toy schema.
- Not a scaling-laws paper. Four points on a curve is a demonstration, not a finding.
- Not novel research. Every component here is standard. The point is that it is *legible*.

## What it is

A complete, honest, end-to-end path from a dataset you can read to a token you can explain —
with a metric, a baseline, and a negative result that wasn't cherry-picked away.

---

*Built from `PLAN.md`. Seed 1337 throughout. Everything reproducible from `python tiny_gpt.py --data`.*
