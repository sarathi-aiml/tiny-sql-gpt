---
title: Tiny SQL GPT
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: A 841K parameter transformer built from scratch, live
---

# Tiny SQL GPT

A 841,216 parameter decoder-only transformer, trained from random weights on a laptop CPU in five
minutes. It writes SQL, and 100% of what it writes executes against a real database.

The vocabulary is 155 tokens, which is the point: small enough to show the entire probability
distribution and every attention head at once. Neither is possible with a frontier model.

- Code: https://github.com/sarathi-aiml/tiny-sql-gpt
- Model: https://huggingface.co/sarathi-balakrishnan/tiny-sql-gpt
