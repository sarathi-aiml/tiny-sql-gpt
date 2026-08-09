"""
Publish the model to the Hugging Face Hub.

Nothing here runs automatically. You need to be logged in first:

    pip install huggingface_hub safetensors
    huggingface-cli login

Then:

    python push_to_hub.py --repo-id YOURNAME/tiny-sql-gpt --dry-run   # look first
    python push_to_hub.py --repo-id YOURNAME/tiny-sql-gpt

The uploaded repo is self-contained: weights, config, model card, and the two
Python files needed to run it. Someone landing on the Hub page can read the
architecture they are about to execute.
"""

import argparse
import os
import shutil

from inference import TinySQLGPT

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "hf", "tiny-sql-gpt")
TEMPLATE = os.path.join(HERE, "hf", "MODEL_CARD.md")

# Uploaded so the Hub repo runs standalone. tiny_gpt.py carries the model
# definition; inference.py is the loader/generator wrapper.
CODE_FILES = ["tiny_gpt.py", "inference.py"]


def stage(repo_id, ckpt):
    """Refresh the staging folder and point the model card at the real repo."""
    model = TinySQLGPT.from_pretrained(ckpt)
    wrote = model.export(STAGE)

    for fn in CODE_FILES:
        shutil.copy2(os.path.join(HERE, fn), os.path.join(STAGE, fn))

    # Render the card from the template every time, so staging is idempotent
    # and re-runnable with any repo id.
    with open(TEMPLATE) as f:
        text = f.read()
    owner = repo_id.split("/")[0]
    text = text.replace("USERNAME/tiny-sql-gpt", repo_id)
    text = text.replace("github.com/USERNAME/", f"github.com/{owner}/")
    with open(os.path.join(STAGE, "README.md"), "w") as f:
        f.write(text)

    files = sorted(os.listdir(STAGE))
    print(f"staged {STAGE}:")
    for fn in files:
        size = os.path.getsize(os.path.join(STAGE, fn))
        print(f"  {fn:<24}{size:>12,} bytes")
    print(f"\nmodel: {model.n_params:,} parameters ({wrote})")
    print(f"sample: {model.generate()}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Publish Tiny SQL GPT to the Hub")
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/tiny-sql-gpt")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "tiny.pt"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and print, upload nothing")
    args = ap.parse_args()

    stage(args.repo_id, args.ckpt)

    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=STAGE, repo_id=args.repo_id, repo_type="model")
    print(f"\nhttps://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
