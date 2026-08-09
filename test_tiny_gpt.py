"""Plain asserts, no framework. Run: python test_tiny_gpt.py"""

import random

import torch

import tiny_gpt as T
import evaluate as E


def test_dataset_is_reproducible():
    a = [T.make_query(random.Random(7)) for _ in range(50)]
    b = [T.make_query(random.Random(7)) for _ in range(50)]
    assert a == b, "same seed must give the same queries"


def test_heldout_pairs_never_grouped_in_training():
    rng = random.Random(T.SEED)
    for _ in range(20_000):
        q = T.make_query(rng).split()
        if "GROUP" not in q:
            continue
        table = q[q.index("FROM") + 1]
        col = q[q.index("GROUP") + 2]
        assert (table, col) not in T.HELD_OUT_GROUPBY, \
            f"held-out pair leaked into training: {table}.{col}"


def test_tokenizer_round_trips():
    rng = random.Random(1)
    qs = [T.make_query(rng) for _ in range(500)]
    tok = T.Tokenizer(qs)
    for q in qs:
        assert tok.decode(tok.encode(q)) == q


def test_param_count_matches_hand_calc():
    cfg = T.Config(vocab_size=100, block_size=64, n_layer=2, n_head=4, n_embd=64)
    m = T.TinyGPT(cfg)
    V, C, L, B = 100, 64, 2, 64
    expected = (
        V * C + B * C                        # token + position embeddings
        + L * (
            (C * 3 * C + 3 * C)              # qkv
            + (C * C + C)                    # attn out proj
            + (C * 4 * C + 4 * C)            # mlp up
            + (4 * C * C + C)                # mlp down
            + 2 * 2 * C                      # two layernorms
        )
        + 2 * C                              # final layernorm
        + C * V                              # lm_head (no bias)
    )
    assert m.n_params() == expected, (m.n_params(), expected)


def test_causal_mask_blocks_the_future():
    """Change a future token; the prediction at position t must not move."""
    cfg = T.Config(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16)
    m = T.TinyGPT(cfg).eval()
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 20
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-6), \
        "future token leaked into earlier positions"


def test_shapes_match_the_diagram():
    cfg = T.Config(vocab_size=110, block_size=64, n_layer=4, n_head=4, n_embd=128)
    m = T.TinyGPT(cfg)
    x = torch.randint(0, 110, (32, 64))
    T.SAVE_ATTN = True
    logits, loss = m(x, x)
    T.SAVE_ATTN = False
    assert logits.shape == (32, 64, 110), logits.shape
    assert loss.item() > 0
    att = m.blocks[0].attn.att_cache
    assert att.shape == (32, 4, 64, 64), att.shape


def test_training_reduces_loss():
    rng = random.Random(3)
    qs = [T.make_query(rng) for _ in range(2000)]
    tok = T.Tokenizer(qs)
    corpus = T.build_corpus(qs, tok)
    cfg = T.Config(vocab_size=len(tok), block_size=32, n_layer=2, n_head=2,
                   n_embd=32)
    splits = T.make_splits(corpus)
    model, hist = T.train(cfg, splits, steps=200, batch_size=32,
                          log_every=200, quiet=True)
    assert hist[-1]["val"] < hist[0]["val"] * 0.6, hist
    return model, tok


def test_log_every_does_not_change_the_trained_model():
    """Evaluation must not consume the training RNG stream.

    estimate_loss() draws batches from the global RNG. Without saving and
    restoring that state, logging more often shifts every later training
    batch, so the same config and seed would train a different model at a
    different log cadence. That silently breaks reproducibility.
    """
    rng = random.Random(11)
    qs = [T.make_query(rng) for _ in range(2000)]
    tok = T.Tokenizer(qs)
    splits = T.make_splits(T.build_corpus(qs, tok))
    mk = lambda: T.Config(vocab_size=len(tok), block_size=32, n_layer=2,
                          n_head=2, n_embd=32)
    a, _ = T.train(mk(), splits, steps=120, batch_size=16, log_every=40, quiet=True)
    b, _ = T.train(mk(), splits, steps=120, batch_size=16, log_every=120, quiet=True)
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(va, vb), f"log_every changed the weights at {k}"


def test_checkpoint_round_trips(tmp="/tmp/_tiny_sql_gpt_test.pt"):
    model, tok = test_training_reduces_loss()
    T.save_ckpt(model, tok, [], tmp)
    m2, t2, _ = T.load_ckpt(tmp)
    assert t2.itos == tok.itos
    x = torch.randint(0, len(tok), (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], m2(x)[0], atol=1e-6)


def test_generation_produces_parseable_output():
    model, tok = test_training_reduces_loss()
    qs = T.sample_queries(model, tok, 5, temperature=0.8)
    assert len(qs) == 5 and all(isinstance(q, str) and q for q in qs)


def test_sqlite_harness_grades_correctly():
    conn = E.build_db()
    assert E.executes(conn, "SELECT region , COUNT ( * ) FROM sales GROUP BY region ;")
    assert not E.executes(conn, "SELECT nope FROM sales ;")
    assert not E.executes(conn, "SELECT FROM WHERE ;")


def test_groupby_rule_detector():
    assert E.groupby_agrees("SELECT region , COUNT ( * ) FROM sales GROUP BY region ;")
    assert not E.groupby_agrees("SELECT region , COUNT ( * ) FROM sales GROUP BY city ;")
    assert E.groupby_agrees("SELECT region FROM sales ;") is None


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
