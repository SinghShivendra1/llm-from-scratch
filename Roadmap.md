# Roadmap

This tracks progress through [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) (the backbone of this repo) plus paper implementations layered in alongside it.

## CS336 — Language Modeling from Scratch

- [x] **1 — Basics: Tokenization** — BPE tokenizer built from scratch (training, encoding, decoding). See [`CS336/1-Tokenization`](./CS336/1-Tokenization).
- [ ] **1 — Basics: Transformer architecture** — attention, feed-forward blocks, positional encoding, assembled from scratch.
- [ ] **1 — Basics: AdamW optimizer + training loop** — train a small language model end-to-end on the tokenizer above.
- [ ] **2 — Systems: Triton GPU kernels** — custom kernels for core ops.
- [ ] **2 — Systems: Distributed data-parallel (DDP) training** — multi-GPU training from scratch.
- [ ] **3 — Scaling: scaling-law experiments** — train small models, extrapolate to predict larger-model behavior.
- [ ] **4 — Data: HTML → text extraction & filtering** — build a data-cleaning pipeline.
- [ ] **4 — Data: MinHash deduplication** — dedupe a training corpus at scale.
- [ ] **5 — Alignment: DPO** — direct preference optimization from scratch.
- [ ] **5 — Alignment: GRPO** — group relative policy optimization from scratch.

## Beyond CS336 — paper implementations (`NLP Lab`)

Loosely ordered by how they build on each other; not strictly sequential.

- [ ] **word2vec** (Mikolov et al.) — skip-gram / CBOW from scratch.
- [ ] **Sequence-to-sequence + attention** (Bahdanau et al.) — the pre-Transformer attention mechanism.
- [ ] **"Attention Is All You Need"** (Vaswani et al.) — full Transformer, independent of the CS336 version, as a cross-check.
- [ ] **BERT** (Devlin et al.) — masked language modeling pretraining, small-scale.
- [ ] **GPT-2** (Radford et al.) — small-scale autoregressive pretraining.
- [ ] **LoRA / PEFT** (Hu et al.) — parameter-efficient fine-tuning on top of a pretrained model.

## Open questions

See [`Research_questions.md`](./Research_questions.md) for things worth digging into further as this progresses — e.g. where BPE's frequency-based merging diverges from linguistically meaningful subword boundaries, and how that shows up differently across languages.

## Notes on process

Each checked-off item should have a short write-up (in its own folder's README or notebook) covering: what was built, what was hard about it, and what I'd do differently. The point of "from scratch" is the understanding, not just a checked box.
