"""
Inference only. No training, no evaluation harness.

This is the file to read if you just want to run the model, and the file the
Hugging Face repo is built around.

    python inference.py                                  # write 5 queries
    python inference.py --prompt "SELECT region ,"       # complete a prompt
    python inference.py --n 10 --temperature 1.2         # sample more, wilder
    python inference.py --run                            # execute them for real
    python inference.py --probs "SELECT region , SUM ( qty ) FROM sales GROUP BY"

    python inference.py --export hf/tiny-sql-gpt         # package for the Hub

Programmatic use:

    from inference import TinySQLGPT
    m = TinySQLGPT.from_pretrained("checkpoints/tiny.pt")
    print(m.generate())
"""

import argparse
import json
import os

import torch
from torch.nn import functional as F

from tiny_gpt import BOS, Config, Tokenizer, TinyGPT, HERE

DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "tiny.pt")


class TinySQLGPT:
    """A trained model plus its tokenizer. Nothing else."""

    def __init__(self, model, tok, device="cpu"):
        self.model, self.tok, self.device = model, tok, device

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, path=DEFAULT_CKPT, device="cpu"):
        """Accepts a .pt checkpoint, a directory in Hub layout, or a repo id."""
        if not os.path.exists(path) and "/" in path and not path.endswith(".pt"):
            from huggingface_hub import snapshot_download   # optional dependency
            path = snapshot_download(repo_id=path)

        if os.path.isdir(path):
            return cls._from_dir(path, device)
        return cls._from_ckpt(path, device)

    @classmethod
    def _from_ckpt(cls, path, device):
        if not os.path.exists(path):
            raise SystemExit(f"No checkpoint at {path}. Run: python tiny_gpt.py --train")
        ck = torch.load(path, map_location=device, weights_only=False)
        return cls._build(ck["cfg"], ck["state_dict"], ck["itos"], device)

    @classmethod
    def _from_dir(cls, path, device):
        """Hub layout: config.json + weights, no pickled Python objects."""
        with open(os.path.join(path, "config.json")) as f:
            meta = json.load(f)
        safe = os.path.join(path, "model.safetensors")
        if os.path.exists(safe):
            from safetensors.torch import load_file
            state = load_file(safe)
        else:
            state = torch.load(os.path.join(path, "pytorch_model.bin"),
                               map_location=device, weights_only=True)
        return cls._build(meta["cfg"], state, meta["itos"], device)

    @classmethod
    def _build(cls, cfg_dict, state, itos, device):
        model = TinyGPT(Config(**cfg_dict)).to(device)
        model.load_state_dict(state)
        model.eval()
        tok = Tokenizer.__new__(Tokenizer)
        tok.itos = itos
        tok.stoi = {s: i for i, s in enumerate(itos)}
        return cls(model, tok, device)

    # ── generation ───────────────────────────────────────────────────────────

    def _ids(self, prompt):
        ids = [self.tok.stoi[BOS]]
        if prompt:
            ids += self.tok.encode(prompt)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def generate(self, prompt="", temperature=0.8, top_k=None, max_new_tokens=None):
        """Return one SQL query as a string."""
        idx = self._ids(prompt)
        room = self.model.cfg.block_size - idx.shape[1]
        out = self.model.generate(
            idx, max_new_tokens=max_new_tokens or room,
            temperature=temperature, top_k=top_k, stop=self.tok.stoi[";"])
        return self.tok.decode(out[0, 1:].tolist())

    def generate_many(self, n=5, **kw):
        return [self.generate(**kw) for _ in range(n)]

    def next_token_probs(self, prompt, top=10):
        """What the model thinks comes next, as (token, probability) pairs.

        The whole distribution is only 155 wide, so `top=None` really does
        return all of it, the thing you cannot do with a frontier model.
        """
        with torch.no_grad():
            logits, _ = self.model(self._ids(prompt))
        probs = F.softmax(logits[0, -1], dim=-1)
        k = top or len(self.tok)
        vals, idx = torch.topk(probs, min(k, len(self.tok)))
        return [(self.tok.itos[i], float(p)) for p, i in zip(vals, idx)]

    @property
    def n_params(self):
        return self.model.n_params()

    # ── export ───────────────────────────────────────────────────────────────

    def export(self, outdir):
        """Write a Hugging Face style folder: config.json + weights.

        Deliberately avoids a pickled checkpoint. Nobody should have to run
        torch.load(weights_only=False) on a stranger's file.
        """
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump({
                "model_type": "tiny-sql-gpt",
                "cfg": {k: v for k, v in vars(self.model.cfg).items()},
                "itos": self.tok.itos,
            }, f, indent=2)
        state = {k: v.contiguous() for k, v in self.model.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(state, os.path.join(outdir, "model.safetensors"))
            wrote = "model.safetensors"
        except ImportError:
            torch.save(state, os.path.join(outdir, "pytorch_model.bin"))
            wrote = "pytorch_model.bin"
        return wrote


def main():
    ap = argparse.ArgumentParser(description="Tiny SQL GPT inference")
    ap.add_argument("--model", default=DEFAULT_CKPT,
                    help=".pt file, Hub-layout directory, or HF repo id")
    ap.add_argument("--prompt", default="", help="text to continue")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--run", action="store_true",
                    help="execute each query against SQLite and report")
    ap.add_argument("--probs", metavar="PROMPT",
                    help="show the next-token distribution for a prompt")
    ap.add_argument("--export", metavar="DIR", help="write a Hub-ready folder")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    m = TinySQLGPT.from_pretrained(args.model, args.device)

    if args.export:
        wrote = m.export(args.export)
        print(f"exported to {args.export}/ (config.json + {wrote})")
        return

    if args.probs:
        print(f"context: {args.probs}\n")
        for t, p in m.next_token_probs(args.probs):
            print(f"  {t:<12} {p:6.3f}  {'█' * int(round(p * 30))}")
        return

    queries = m.generate_many(args.n, prompt=args.prompt,
                              temperature=args.temperature, top_k=args.top_k)
    if not args.run:
        print("\n".join(queries))
        return

    import evaluate                      # only needed for --run
    conn = evaluate.build_db()
    ok = 0
    for q in queries:
        good = evaluate.executes(conn, q)
        ok += good
        print(f"  [{'OK  ' if good else 'FAIL'}] {q}")
    print(f"\n{ok}/{len(queries)} executed against a real database "
          f"({m.n_params:,} parameters)")


if __name__ == "__main__":
    main()
