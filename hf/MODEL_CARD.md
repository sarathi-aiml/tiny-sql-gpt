---
license: mit
language:
  - en
tags:
  - text-generation
  - sql
  - educational
  - from-scratch
  - interpretability
  - tiny
pipeline_tag: text-generation
library_name: pytorch
---

# Tiny SQL GPT

**A 841,216-parameter decoder-only transformer, trained from random weights on a laptop CPU in
five minutes. 100% of the SQL it generates executes against a real database.**

No pretrained weights. No `transformers` model classes. No API keys. The full architecture is
~200 readable lines of PyTorch.

This model exists to be **understood**, not deployed. Its vocabulary is 155 tokens, which is the
point: small enough that you can print the *entire* probability distribution at every generation
step, something no frontier model demo can do.

- **Code, evaluation harness and write-up:** https://github.com/USERNAME/tiny-sql-gpt
- **Plain-English explainer (no maths):** [`EXPLAIN.md`](https://github.com/USERNAME/tiny-sql-gpt/blob/main/EXPLAIN.md)

---

## Usage

```bash
pip install torch safetensors huggingface_hub
```

```python
from inference import TinySQLGPT     # inference.py + tiny_gpt.py from the repo

model = TinySQLGPT.from_pretrained("USERNAME/tiny-sql-gpt")

print(model.generate())
# SELECT segment , AVG ( qty ) FROM sales WHERE product = 'cog' GROUP BY segment ;

print(model.generate(prompt="SELECT region ,"))

# The whole distribution, all 155 tokens, not a top-k truncation
for token, p in model.next_token_probs(
        "SELECT region , SUM ( qty ) FROM sales GROUP BY", top=5):
    print(f"{token:<12} {p:.3f}")
# region       0.995
# segment      0.001
# quarter      0.001
```

---

## What it does

Generates SQL over a fixed three-table schema (`sales`, `customers`, `orders`) using 14 query
shapes: `SELECT`, `WHERE`, `AND`, `GROUP BY`, `ORDER BY`, `LIMIT`, and the aggregates
`COUNT`/`SUM`/`AVG`/`MAX`/`MIN`.

It is **not** a text-to-SQL model. It does not take a natural-language question. It generates
SQL unconditionally, or continues a SQL prefix you give it.

## Results

500 generated queries, executed against a real SQLite database. Seeded, so you get these exact
numbers.

| metric | Tiny SQL GPT | bigram baseline |
|---|---:|---:|
| executes | **100.0%** | 4.4% |
| `GROUP BY` agrees with `SELECT` | **100.0%** | 3.4% |
| novel (not in training set) | 13.4% | 98.6% |
| validation loss | 0.663 | n/a |

100% is a real measurement, but read it against the task: 14 query shapes, 3 tables, 155 tokens.
A model that saturates *this* is proof the training loop works, not a text-to-SQL system.

### Scaling

Same architecture and data at four sizes, all trained on one laptop:

| model | params | executes | `GROUP BY` agrees |
|---|---:|---:|---:|
| nano | 24,736 | 99.6% | **41.4%** |
| micro | 124,032 | 99.8% | **100.0%** |
| **tiny** (this model) | **841,216** | **100.0%** | **100.0%** |
| small | 4,834,816 | 100.0% | 100.0% |

Syntax is nearly free: 24K parameters writes SQL that runs. The long-range dependency costs ~5x
more, and appears as a phase transition between 24K and 125K. Above that, nothing improves: all
four converge to ~0.66 validation loss, which is the entropy of the data generator, not a limit
of the models.

---

## Why it's interesting

The training data contains a deliberately planted long-range dependency: **the column after
`GROUP BY` is always the column that appeared first in `SELECT`**, roughly 8 tokens earlier.
Getting it right requires looking back, which is what attention is for.

**An attention head learned it.** Probing all 16 heads, layer 1 head 1 places **92.5% of its
attention** on the `SELECT` column, 11.1x above uniform. Nobody designed or labelled that head.

**And you can watch it hallucinate.** Three `(table, column)` pairs were held out from the
`GROUP BY` position during training. The columns appear elsewhere, just never there. The model
gets **0 out of 3**, confidently substituting a familiar column instead:

```
asked for:   ... GROUP BY  ->  "channel"    (never seen in this position)
it answered: ... GROUP BY  ->  "carrier"    (familiar, confident, wrong)
```

A control prompt shows the copy circuit *is* firing (2.6x to 13.9x lift), it simply loses to a prior
against tokens never seen in that slot. **Attention identifies the right source token; the output
prior overrules it.** That is hallucination, in a model small enough to point at the exact cause.

---

## Training

| | |
|---|---|
| data | 100,000 generated SQL queries (16,941 unique), 1.3M tokens, seed 1337 |
| architecture | decoder-only, 4 layers, 4 heads, 128 embedding, 64-token context |
| tokenizer | word-level, 155 tokens |
| optimizer | AdamW, lr 1e-3, cosine schedule, weight decay 0.01, grad clip 1.0 |
| steps | 3,000 · batch 64 · ~5 minutes on a laptop CPU |
| final loss | train 0.657 · val 0.663 |

Training data is **generated, not scraped**, from a grammar in the repo. No licensing questions,
and a learner can read the entire source of the training set.

## Limitations

- Not text-to-SQL. No natural-language input.
- One fixed three-table schema. It knows no other tables or columns.
- 64-token context. Longer queries are truncated.
- Does not generalize to column names it never saw in a given syntactic position (see above,
  that failure is the point, and it is measured rather than hidden).
- Generated SQL is syntactically valid, not semantically meaningful. It will happily write
  `WHERE age > 1500`.

## License

MIT.
