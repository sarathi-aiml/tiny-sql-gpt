"""
Tiny SQL GPT, live.

A Gradio demo for a 841,216 parameter transformer trained from random weights.
Three things you cannot do with a frontier model:

  1. generate text and immediately execute it to check correctness
  2. see the ENTIRE probability distribution, all 155 tokens, not a top-k slice
  3. look at every attention head and find the one doing a specific job

Weights and model code are pulled from the Hub repo, so this Space has exactly
one source of truth.
"""

import sqlite3
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download

MODEL_REPO = "sarathi-balakrishnan/tiny-sql-gpt"

MODEL_DIR = snapshot_download(MODEL_REPO)
sys.path.insert(0, MODEL_DIR)

import tiny_gpt as T                      # noqa: E402
from inference import TinySQLGPT          # noqa: E402

MODEL = TinySQLGPT.from_pretrained(MODEL_DIR)
TOK = MODEL.tok
PARAMS = f"{MODEL.n_params:,}"


# ── a real database, so "does it work" is not a matter of opinion ────────────

def build_db(rows=200, seed=T.SEED):
    import random
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for table, cols in T.SCHEMA.items():
        decl = ", ".join([f"{c} TEXT" for c in cols["cat"]] +
                         [f"{c} REAL" for c in cols["num"]])
        conn.execute(f"CREATE TABLE {table} ({decl})")
        n = len(cols["cat"]) + len(cols["num"])
        conn.executemany(
            f"INSERT INTO {table} VALUES ({','.join('?' * n)})",
            [tuple([rng.choice(T.VALUES[c]).strip("'") for c in cols["cat"]] +
                   [rng.randint(1, 2000) for _ in cols["num"]]) for _ in range(rows)])
    conn.commit()
    return conn


DB = build_db()


def run_sql(q):
    try:
        rows = DB.execute(q).fetchall()
        return True, f"{len(rows)} row(s)"
    except sqlite3.Error as e:
        return False, str(e)[:60]


# ── 1. generate, then actually run it ───────────────────────────────────────

def do_generate(n, temperature, top_k, prompt):
    top_k = int(top_k) if top_k and int(top_k) > 0 else None
    lines = ["| | query | result |", "|---|---|---|"]
    ok = 0
    for _ in range(int(n)):
        q = MODEL.generate(prompt=prompt.strip(), temperature=temperature, top_k=top_k)
        good, detail = run_sql(q)
        ok += good
        lines.append(f"| {'PASS' if good else 'FAIL'} | `{q}` | {detail} |")
    header = (f"**{ok}/{int(n)} executed** against a live SQLite database, "
              f"from a {PARAMS} parameter model.\n\n")
    return header + "\n".join(lines)


# ── 2. the entire distribution ──────────────────────────────────────────────

def do_distribution(prompt, temperature):
    prompt = prompt.strip()
    try:
        ids = [TOK.stoi[t] for t in prompt.split()]
    except KeyError as e:
        return f"Token {e} is not in the 155 token vocabulary.", None
    if not ids:
        return "Type a SQL prefix first.", None

    x = torch.tensor([[TOK.stoi[T.BOS]] + ids])
    with torch.no_grad():
        logits, _ = MODEL.model(x)
    row = logits[0, -1]
    probs = torch.softmax(row / max(temperature, 1e-6), dim=-1)

    top = torch.topk(probs, 10)
    md = [f"Next token after `{prompt}`", "", "| token | probability | |", "|---|---:|---|"]
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        md.append(f"| `{TOK.itos[i]}` | {p:.4f} | {'█' * int(round(p * 40))} |")
    tail = 1 - top.values.sum().item()
    md.append(f"| *(remaining {len(TOK) - 10} tokens)* | {tail:.4f} | |")
    md.append("")
    md.append(f"That is the **complete** distribution. All {len(TOK)} tokens sum to 1.0. "
              f"Raise the temperature and watch probability mass move from the "
              f"best answer to worse ones.")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(probs)), probs.tolist(), width=1.0)
    ax.set_xlabel(f"token id (all {len(TOK)})")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"full softmax at temperature {temperature}")
    fig.tight_layout()
    return "\n".join(md), fig


# ── 3. every attention head ─────────────────────────────────────────────────

def do_attention(prompt):
    prompt = prompt.strip()
    toks = [T.BOS] + prompt.split()
    try:
        ids = torch.tensor([[TOK.stoi[t] for t in toks]])
    except KeyError as e:
        return f"Token {e} is not in the 155 token vocabulary.", None

    T.SAVE_ATTN = True
    with torch.no_grad():
        MODEL.model(ids)
    T.SAVE_ATTN = False

    rows = []
    for li, blk in enumerate(MODEL.model.blocks):
        att = blk.attn.att_cache
        for h in range(att.shape[1]):
            rows.append((li, h, att[0, h]))

    target = 2 if len(toks) > 2 else 1        # the SELECT column position
    scored = sorted(rows, key=lambda r: float(r[2][-1, target]), reverse=True)
    uniform = 1.0 / len(toks)

    md = [f"Where each head looks when predicting the token after "
          f"`{prompt.split()[-1]}`.", "",
          f"Scored on attention paid to position {target} "
          f"(`{toks[target]}`). Uniform would be {uniform:.1%}.", "",
          "| head | attention on that token | |", "|---|---:|---|"]
    for li, h, att in scored[:8]:
        m = float(att[-1, target])
        md.append(f"| L{li}H{h} | {m:.3f} | {'█' * int(round(m * 40))} |")

    bl, bh, best = scored[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(best.numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=8)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=8)
    ax.set_xlabel("attending to"); ax.set_ylabel("token at position")
    ax.set_title(f"layer {bl}, head {bh}\nblank upper triangle: it cannot see the future")
    fig.tight_layout()
    return "\n".join(md), fig


# ── UI ──────────────────────────────────────────────────────────────────────

INTRO = f"""
# Tiny SQL GPT

**The goal:** build a language model from scratch, end to end, small enough to understand every
part of it, and in doing so understand how the large ones actually work.

This is a **{PARAMS} parameter** transformer trained from random weights on a laptop CPU in five
minutes. No pretrained weights, no `transformers` library, no API. It writes SQL over a fixed
three-table schema, and 100% of what it writes executes.

Its vocabulary is only {len(TOK)} tokens, which is the whole point: small enough that you can see
the entire probability distribution and every attention head at once.

[Code on GitHub](https://github.com/sarathi-aiml/tiny-sql-gpt) ·
[Model on the Hub](https://huggingface.co/{MODEL_REPO}) ·
[Plain English explainer](https://github.com/sarathi-aiml/tiny-sql-gpt/blob/main/EXPLAIN.md)
"""

with gr.Blocks(title="Tiny SQL GPT") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("1. Write SQL, then run it"):
        gr.Markdown("The model generates SQL and every statement is immediately executed "
                    "against a real SQLite database. Nothing is graded by eye.")
        with gr.Row():
            g_n = gr.Slider(1, 10, value=5, step=1, label="how many")
            g_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="temperature")
            g_topk = gr.Slider(0, 50, value=0, step=1, label="top-k (0 = off)")
        g_prompt = gr.Textbox(label="optional prefix to continue", placeholder="SELECT region ,")
        g_btn = gr.Button("Generate and execute", variant="primary")
        g_out = gr.Markdown()
        g_btn.click(do_generate, [g_n, g_temp, g_topk, g_prompt], g_out)

    with gr.Tab("2. The entire probability distribution"):
        gr.Markdown("A frontier model has 100,000+ tokens in its vocabulary, so nobody can show "
                    "you its full output distribution. This one has 155. Here is all of it.\n\n"
                    "Try a constrained position (`... GROUP BY`) against an open one (`SELECT`) "
                    "and watch confidence change with context, not with the model.")
        d_prompt = gr.Textbox(value="SELECT region , SUM ( qty ) FROM sales GROUP BY",
                              label="SQL prefix (tokens separated by spaces)")
        d_temp = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="temperature")
        d_btn = gr.Button("Show the distribution", variant="primary")
        d_out = gr.Markdown()
        d_plot = gr.Plot()
        d_btn.click(do_distribution, [d_prompt, d_temp], [d_out, d_plot])

    with gr.Tab("3. Every attention head"):
        gr.Markdown("The training data has a rule planted in it: the column after `GROUP BY` is "
                    "always the column that appeared first in `SELECT`, about 8 tokens earlier. "
                    "Getting that right requires looking back.\n\n"
                    "All 16 heads are scored below on how much attention they pay to the "
                    "`SELECT` column. One of them learned the job. Nobody programmed it.")
        a_prompt = gr.Textbox(value="SELECT region , SUM ( qty ) FROM sales GROUP BY",
                              label="SQL prefix")
        a_btn = gr.Button("Probe all 16 heads", variant="primary")
        a_out = gr.Markdown()
        a_plot = gr.Plot()
        a_btn.click(do_attention, [a_prompt], [a_out, a_plot])

    gr.Markdown("*Fiction-free: every number here is computed live by the model in this Space.*")

if __name__ == "__main__":
    demo.launch()
