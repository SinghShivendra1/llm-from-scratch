# LLM-Playground

Learning how large language models work by building the core pieces from scratch — starting with Stanford's [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/), then extending into individual paper implementations and experiments beyond the course.

The goal isn't to use `transformers.AutoModel` and call it done — it's to understand every layer well enough to have built it myself: tokenizer, attention, training loop, and the systems/data/alignment work around them.

## Why this repo

Most tutorials let you copy-paste your way to a working model without understanding *why* any of it works. This repo is the opposite: everything here is implemented from first principles, with notes on what I learned and where I got stuck, so the process is as visible as the result.

## What's implemented so far

- **[CS336 Assignment 1 — Tokenization](./CS336/1-Tokenization)**: a byte-pair encoding (BPE) tokenizer built from scratch — training (learning merges from a corpus), encoding, and decoding, with no external tokenizer libraries used.

See [`Roadmap.md`](./Roadmap.md) for what's done, in progress, and coming next across the rest of CS336 and beyond.

## How to run

Each component lives in its own folder with its own instructions. For the tokenizer:

```bash
cd CS336/1-Tokenization
# see the folder's own notes/notebook for training + encode/decode usage
```

(As more components land, this section will list a run command for each — the near-term goal is a consistent `python train.py` / `python run.py` pattern across folders.)

## Repository structure

```
LLM-Playground/
├── CS336/
│   └── 1-Tokenization/     # BPE tokenizer (done)
│       # 2-Systems, 3-Scaling, 4-Data, 5-Alignment to follow
├── NLP Lab/                # standalone paper implementations / experiments
├── Research_questions.md   # open questions and things worth digging into
├── Roadmap.md              # what's done, in progress, and next
└── README.md
```

## Roadmap

Full detail in [`Roadmap.md`](./Roadmap.md) — short version: CS336's five assignments (Basics, Systems, Scaling, Data, Alignment) form the backbone, with classic paper implementations (word2vec, attention, Transformer, BERT-style pretraining) layered in along the way.

## License

MIT — see [`LICENSE`](./LICENSE).
