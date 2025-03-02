# HashHop Attention

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## An Experimental Approach for Long-Context Processing

HashHop Attention is an approach we're exploring for handling extremely long sequences (up to 100M tokens) in transformer models. It introduces a "hopping" mechanism that allows models to efficiently navigate through large contexts, similar to how humans might use an index to find information in a long document.

## Research 

We're investigating three key technical challenges that current language models face with very long texts:

1. **Computational Scaling**: Traditional attention mechanisms become prohibitively expensive due to its quadratic complexity as context length increases. We're testing whether HashHop's selective attention pattern can provide better computational efficiency.

2. **Memory Management**: Standard models require enormous memory to store KV cache for millions of tokens. Our experiments explore whether HashHop's hopping mechanism can process longer sequences with fixed memory constraints.

3. **Long-Range Dependencies**: Models often struggle to connect information separated by large distances. We're evaluating if HashHop can maintain reliable information retrieval across very long ranges.

These challenges are particularly relevant for real-world applications involving large documents or codebases, where having the full context available could significantly improve model performance.

## Evaluation Framework

Current evaluation methods have problems. For example, the "Needle in a Haystack" test places a random fact in a long document and asks the model to find it. But unusual facts naturally stand out in context (like "Max and Arun having coffee" inserted into a novel about whales), making the test easier than real-world tasks.

HashHop uses random strings (called hashes) that don't have any inherent meaning or patterns. For example:
```
jJWlupoT → KmsFrnRa
vRLWdcwV → sVLdzfJu
YOJVrdjK → WKPUyWON
```

By using these meaningless strings, we force models to truly store and retrieve information rather than relying on patterns or context clues.

## How HashHop Works

HashHop introduces three key ideas:

1. The model learns which parts of the text are important reference points.
2. Instead of trying to connect everything at once, the model follows chains of references in steps.
3. Special components transform information between steps to help the model follow connections.

Think of it like using a library index to find information in a book, rather than reading every page sequentially.

## Technical Approach

At its core, HashHop Attention works by:

1. **Identifying Reference Points**: Using "hash gates" to determine which tokens serve as pointers to other information
2. **Following Connections**: Explicitly calculating multiple steps of attention to follow chains of references
3. **Transforming Information**: Using projection layers to transform intermediate results between hops

```
┌─────────────────────────────────────────────┐
│             HashHopTransformer              │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │              Embedding Layer            │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │           Positional Encoding           │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │            HashHopBlock × N             │ │
│ │ ┌───────────────────────────────────┐   │ │
│ │ │         HashHopAttention          │   │ │
│ │ │  ┌─────────────┐ ┌─────────────┐  │   │ │
│ │ │  │  Hash Gate  │ │Hash Project │  │   │ │
│ │ │  └─────────────┘ └─────────────┘  │   │ │
│ │ └───────────────────────────────────┘   │ │
│ │ ┌───────────────────────────────────┐   │ │
│ │ │           Feed Forward            │   │ │
│ │ └───────────────────────────────────┘   │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │            Projection Layer             │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Repository Structure

- `model.py`: Core HashHop architecture implementation
- `transformer_model.py`: Base transformer components
- `dataset.py`: HashHop benchmark dataset implementation
- `train.py`: Training procedures for HashHop models
- `evals/`: Evaluation tools and metrics

## Benchmark Tasks

The HashHop benchmark tests a model's ability to:

1. **Single-Hop Retrieval**: Find matching values for random hash keys
2. **Multi-Hop Chaining**: Follow chains of hash references (Hash1 → Hash2 → Hash3 → ...)
3. **Position-Invariant Reasoning**: Process hash pairs regardless of position in context

Hashes are random and incompressible, requiring the model to actually store and retrieve the maximum possible information content, avoiding any polysemanticity.

Example data looks like this:
```
jJWlupoT → KmsFrnRa
vRLWdcwV → sVLdzfJu
YOJVrdjK → WKPUyWON
```
## Getting Started

### Installation

```bash

git clone https://github.com/theharshithh/hash-hop-attn.git
cd hash-hop-attn

uv venv; source .venv/bin/activate

uv pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Evaluation

```bash
python val.py
```

## This repository is a work in progress 👷

## Further Goals

This is an initial implementation inspired by the HashHop attention mechanism described by [Magic.dev](https://magic.dev/blog/100m-token-context-windows). Our goal is to see if this approach can scale to ultra-long contexts (>100M tokens).

Further roadmap includes:
- Multi-scale hash indexing
- Chunking and routing mechanisms
- Distributed context processing
- Memory optimization techniques

## Citation

If you use HashHop in your research, please cite:

```
@software{hashhop2024,
  author = {Harshith Murthy},
  title = {HashHop Attention for Ultra-Long Context Processing},
  url = {https://github.com/theharshithh/hash-hop-attn},
  year = {2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
