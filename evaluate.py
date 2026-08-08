"""
The part that makes this more than a tutorial.

Three artifacts almost nobody ships with a from-scratch transformer:

    --eval        % of generated SQL that actually EXECUTES, vs a bigram baseline
    --scaling     the same architecture at 4 sizes, one chart, one laptop
    --attention   probe every head for the GROUP BY -> SELECT dependency

    python evaluate.py --eval
    python evaluate.py --scaling
    python evaluate.py --attention
"""

import argparse
import json
import os
import sqlite3

import torch

import tiny_gpt as T
from tiny_gpt import (BOS, Bigram, Config, SIZES, Tokenizer, build_corpus,
                      load_ckpt, load_queries, make_splits, sample_queries, train)

FIG_DIR = os.path.join(T.HERE, "figures")
N_SAMPLES = 500


# ─────────────────────────────────────────────────────────────────────────────
# A real database. Generated SQL is run against it. No grading on vibes.
# ─────────────────────────────────────────────────────────────────────────────

def build_db(seed=T.SEED, rows=200):
    import random
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    types = {"cat": "TEXT", "num": "REAL"}
    for table, cols in T.SCHEMA.items():
        decl = ", ".join(
            [f"{c} {types['cat']}" for c in cols["cat"]] +
            [f"{c} {types['num']}" for c in cols["num"]]
        )
        conn.execute(f"CREATE TABLE {table} ({decl})")
        ncols = len(cols["cat"]) + len(cols["num"])
        data = [
            tuple([rng.choice(T.VALUES[c]).strip("'") for c in cols["cat"]] +
                  [rng.randint(1, 2000) for _ in cols["num"]])
            for _ in range(rows)
        ]
        conn.executemany(
            f"INSERT INTO {table} VALUES ({','.join('?' * ncols)})", data
        )
    conn.commit()
    return conn


def parses(conn, q):
    try:
        conn.execute("EXPLAIN " + q)
        return True
    except sqlite3.Error:
        return False


def executes(conn, q):
    try:
        conn.execute(q).fetchall()
        return True
    except sqlite3.Error:
        return False


def groupby_agrees(q):
    """Returns None if the query has no GROUP BY, else True/False.

    The rule the training data always follows: the column after GROUP BY is
    the same column that appears first in SELECT. Learning this requires
    looking back ~8 tokens — which is exactly what attention is for.
    """
    toks = q.split()
    if "GROUP" not in toks:
        return None
    i = toks.index("GROUP")
    if i + 2 >= len(toks) or len(toks) < 2:
        return False
    return toks[1] == toks[i + 2]


# ─────────────────────────────────────────────────────────────────────────────
# The headline number
# ─────────────────────────────────────────────────────────────────────────────

def score(queries, conn, train_set):
    n = len(queries)
    if n == 0:
        return {}
    gb = [groupby_agrees(q) for q in queries]
    gb_seen = [g for g in gb if g is not None]
    return {
        "n": n,
        "parses_pct": 100 * sum(parses(conn, q) for q in queries) / n,
        "executes_pct": 100 * sum(executes(conn, q) for q in queries) / n,
        "has_groupby_pct": 100 * len(gb_seen) / n,
        "groupby_agrees_pct": (100 * sum(gb_seen) / len(gb_seen)) if gb_seen else 0.0,
        "novel_pct": 100 * sum(q not in train_set for q in queries) / n,
    }


def bigram_queries(tok, corpus, n, device="cpu"):
    bg = Bigram(len(tok)).fit(corpus)
    bos, semi = tok.stoi[BOS], tok.stoi[";"]
    out = []
    for _ in range(n):
        ids = bg.generate(bos, max_new_tokens=40, stop=semi)
        out.append(tok.decode(ids[1:]))
    return out


def _next_token_probs(model, tok, prompt, device):
    ids = torch.tensor([[tok.stoi[BOS]] + tok.encode(prompt)],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(ids)
    return torch.softmax(logits[0, -1], dim=-1)


def heldout_probe(model, tok, device="cpu"):
    """Did it learn the RULE ("copy the SELECT column") or the PAIRS?

    These (table, column) pairs never appeared in a GROUP BY during training.
    We prompt up to `GROUP BY` and ask what it wants to say next.

    A bare pass/fail is not enough. If it says the wrong column, we still want
    to know whether the copy circuit fired at all — so we run a CONTROL prompt
    with a different SELECT column and compare. The ratio isolates the effect
    of the SELECT column from the model's blanket prior over tokens:

        lift = p(col | SELECT col ... GROUP BY) / p(col | SELECT other ... GROUP BY)

    lift >> 1 means context IS being used, but is losing to the prior.
    That is hallucination in miniature, on 841K parameters.
    """
    rows = []
    for table, col in T.HELD_OUT_GROUPBY:
        seen = [c for c in T.SCHEMA[table]["cat"]
                if (table, c) not in T.HELD_OUT_GROUPBY]
        probs = _next_token_probs(
            model, tok, f"SELECT {col} , COUNT ( * ) FROM {table} GROUP BY", device)
        ctrl = _next_token_probs(
            model, tok,
            f"SELECT {seen[0]} , COUNT ( * ) FROM {table} GROUP BY", device)

        p = float(probs[tok.stoi[col]])
        p_ctrl = float(ctrl[tok.stoi[col]])
        rank = int((probs > probs[tok.stoi[col]]).sum()) + 1
        rows.append({
            "table": table, "column": col,
            "predicted": tok.itos[int(probs.argmax())],
            "correct": tok.itos[int(probs.argmax())] == col,
            "p_correct": p, "p_control": p_ctrl,
            "lift": p / max(p_ctrl, 1e-9), "rank": rank, "vocab": len(tok),
        })
    return rows


def run_eval(size="tiny", device="cpu", n=N_SAMPLES, quiet=False):
    ckpt = os.path.join(T.CKPT_DIR, f"{size}.pt")
    model, tok, _ = load_ckpt(ckpt, device)
    queries = load_queries()
    train_set = set(queries)
    corpus = build_corpus(queries, tok)
    conn = build_db()

    gpt_q = sample_queries(model, tok, n, temperature=0.8, device=device)
    bg_q = bigram_queries(tok, corpus, n, device)

    res = {
        "size": size,
        "params": model.n_params(),
        "gpt": score(gpt_q, conn, train_set),
        "bigram": score(bg_q, conn, train_set),
        "heldout": heldout_probe(model, tok, device),
    }
    if not quiet:
        report(res, gpt_q, bg_q)
    return res


def report(res, gpt_q, bg_q):
    g, b = res["gpt"], res["bigram"]
    print("\n" + "=" * 68)
    print(f"EXECUTABLE EVAL — {g['n']} generated queries, run against real SQLite")
    print("=" * 68)
    print(f"{'metric':<26}{'TinyGPT':>12}{'bigram':>12}")
    print(f"{'params':<26}{res['params']:>12,}{'—':>12}")
    print("-" * 50)
    for k, label in [("parses_pct", "parses"),
                     ("executes_pct", "EXECUTES"),
                     ("has_groupby_pct", "has GROUP BY"),
                     ("groupby_agrees_pct", "GROUP BY agrees"),
                     ("novel_pct", "novel (not in train)")]:
        print(f"{label:<26}{g[k]:>11.1f}%{b[k]:>11.1f}%")

    print("\n" + "=" * 68)
    print("GENERALIZATION — (table, column) pairs never grouped during training")
    print("=" * 68)
    for r in res["heldout"]:
        mark = "OK " if r["correct"] else "MISS"
        print(f"  [{mark}] SELECT {r['column']} ... FROM {r['table']} GROUP BY "
              f"-> {r['predicted']}")
        print(f"         p({r['column']})={r['p_correct']:.4f}  "
              f"rank {r['rank']}/{r['vocab']}  |  control "
              f"p={r['p_control']:.6f}  ->  copy lift {r['lift']:>6.1f}x")
    hits = sum(r["correct"] for r in res["heldout"])
    lift = sum(r["lift"] for r in res["heldout"]) / len(res["heldout"])
    print(f"\n  {hits}/{len(res['heldout'])} correct on unseen pairs")
    print(f"  mean copy lift: {lift:.1f}x — naming the column in SELECT raises")
    print(f"  its GROUP BY probability by this much, even when it still loses.")

    print("\n" + "=" * 68)
    print("SAMPLES")
    print("=" * 68)
    print("TinyGPT:")
    for q in gpt_q[:5]:
        print(f"  {q}")
    print("bigram:")
    for q in bg_q[:3]:
        print(f"  {q}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# The scaling curve — same data, same code, four sizes, one laptop
# ─────────────────────────────────────────────────────────────────────────────

def run_scaling(steps=2000, device="cpu", sizes=None):
    queries = load_queries()
    tok = Tokenizer(queries)
    corpus = build_corpus(queries, tok)
    splits = make_splits(corpus)
    results = []

    for name in (sizes or list(SIZES)):
        cfg = Config(name=name, vocab_size=len(tok), **SIZES[name])
        model, hist = train(cfg, splits, steps=steps, device=device,
                            log_every=max(steps // 2, 1))
        T.save_ckpt(model, tok, hist, os.path.join(T.CKPT_DIR, f"{name}.pt"))
        r = run_eval(name, device, quiet=True)
        results.append({
            "name": name, "params": model.n_params(),
            "val_loss": hist[-1]["val"],
            "executes_pct": r["gpt"]["executes_pct"],
            "groupby_agrees_pct": r["gpt"]["groupby_agrees_pct"],
            "heldout_correct": sum(x["correct"] for x in r["heldout"]),
        })
        print(f"  -> {r['gpt']['executes_pct']:.1f}% executable\n")

    print("=" * 68)
    print("SCALING — same data, same code, four model sizes")
    print("=" * 68)
    print(f"{'model':<9}{'params':>11}{'val loss':>11}{'executes':>11}"
          f"{'GB agrees':>11}{'heldout':>9}")
    for r in results:
        print(f"{r['name']:<9}{r['params']:>11,}{r['val_loss']:>11.3f}"
              f"{r['executes_pct']:>10.1f}%{r['groupby_agrees_pct']:>10.1f}%"
              f"{r['heldout_correct']:>7}/3")

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_scaling(results)
    return results


def plot_scaling(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping chart)")
        return
    x = [r["params"] for r in results]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, [r["executes_pct"] for r in results], "o-", lw=2)
    ax[0].set_xscale("log"); ax[0].set_ylim(-5, 105)
    ax[0].set_xlabel("parameters"); ax[0].set_ylabel("% of generated SQL that runs")
    ax[0].set_title("Bigger model, more valid SQL")
    ax[1].plot(x, [r["val_loss"] for r in results], "o-", lw=2, color="crimson")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("parameters"); ax[1].set_ylabel("validation loss")
    ax[1].set_title("Loss vs scale")
    for a in ax:
        a.grid(alpha=.3)
        for r in results:
            a.annotate(r["name"], (r["params"], 0), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=8, alpha=.7)
    fig.suptitle("Tiny SQL GPT — trained from random weights on a laptop")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "scaling.png")
    fig.savefig(out, dpi=150)
    print(f"\nchart -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Attention probe — is there a head that learned GROUP BY agreement?
# ─────────────────────────────────────────────────────────────────────────────

def run_attention(size="tiny", device="cpu"):
    model, tok, _ = load_ckpt(os.path.join(T.CKPT_DIR, f"{size}.pt"), device)
    prompt = "SELECT region , SUM ( qty ) FROM sales GROUP BY"
    toks = [BOS] + prompt.split()
    ids = torch.tensor([[tok.stoi[t] for t in toks]], dtype=torch.long,
                       device=device)

    T.SAVE_ATTN = True
    with torch.no_grad():
        model(ids)
    T.SAVE_ATTN = False

    target = toks.index("region")          # the SELECT column position
    print("\n" + "=" * 68)
    print("ATTENTION PROBE — GROUP BY -> SELECT agreement")
    print("=" * 68)
    print(f"query    : {prompt}")
    print(f"question : predicting the token after GROUP BY, does any head look")
    print(f"           back at position {target} ('region', the SELECT column)?\n")

    rows = []
    for li, blk in enumerate(model.blocks):
        att = blk.attn.att_cache
        for h in range(att.shape[1]):
            mass = float(att[0, h, -1, target])
            rows.append((mass, li, h, att[0, h].cpu()))
    rows.sort(reverse=True, key=lambda r: r[0])

    for mass, li, h, _ in rows:
        bar = "█" * int(round(mass * 40))
        print(f"  L{li}H{h}   {mass:5.3f}  {bar}")

    best_mass, bl, bh, best_att = rows[0]
    print(f"\n  best: layer {bl}, head {bh} — {best_mass:.1%} of its attention")
    print(f"        from the final position lands on the SELECT column.")
    uniform = 1.0 / len(toks)
    print(f"  uniform baseline would be {uniform:.1%} "
          f"({best_mass / uniform:.1f}x)")
    print("\n  VERDICT: " + (
        "a head learned the GROUP BY dependency."
        if best_mass > 3 * uniform else
        "no single head owns this dependency — it is distributed. "
        "Honest negative; the textbook diagram is cleaner than reality."))
    plot_attention(best_att, toks, bl, bh)
    return rows


def plot_attention(att, toks, layer, head):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(att.numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=8)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=8)
    ax.set_xlabel("attending to"); ax.set_ylabel("token at position")
    ax.set_title(f"Tiny SQL GPT — layer {layer}, head {head}\n"
                 f"lower triangle only: it cannot see the future")
    fig.colorbar(im, shrink=.8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "attention.png")
    fig.savefig(out, dpi=150)
    print(f"  heatmap -> {out}\n")


def main():
    ap = argparse.ArgumentParser(description="Tiny SQL GPT — evaluation")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--size", default="tiny", choices=list(SIZES))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = T.pick_device(args.device)

    if args.eval:
        run_eval(args.size, device, args.n)
    elif args.scaling:
        run_scaling(args.steps, device)
    elif args.attention:
        run_attention(args.size, device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
