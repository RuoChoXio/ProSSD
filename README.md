# Verifiable LLM-Generated Text Detection via Projected Semantic-Structural Distributions

**ACL 2026 (Long Papers)** | [ACL Anthology](https://aclanthology.org/2026.acl-long.638/) | [Paper PDF](https://aclanthology.org/2026.acl-long.638.pdf)

This repository provides the implementation and datasets for **Verifiable LLM-Generated Text Detection via Projected Semantic-Structural Distributions**.

## Paper overview

Detecting LLM-generated text requires reliable statistical evidence that can be inspected and verified. Existing approaches based on proxy-model output probabilities or semantic features alone can suffer from distribution mismatch and limited interpretability. **ProSSD** models the joint semantic-structural distributions of human-written and machine-generated text, motivated by the observation that machine-generated text exhibits a systematic directional shift under comparable syntactic structures.

The framework has three components:

1. **Supervised subspace projection:** learn a compact, label-aware projection using partial least squares (PLS) on embeddings from a frozen pretrained encoder.
2. **Semantic-structural distribution modeling:** group projected local features by part-of-speech patterns and fit separate conditional Gaussian distributions for human and machine text.
3. **Wasserstein-weighted detection:** aggregate modified Mahalanobis-distance differences derived from a likelihood-ratio test, weighting structural patterns by their distributional separation.

The paper investigates cross-domain, cross-model, and adversarial detection, data efficiency, and computational efficiency. It also examines systematic semantic translation and semantic collapse to provide interpretable statistical evidence about LLM generation behavior. This release contains the core detection pipeline, domain/model evaluation data, and training-size subsets; it does not include scripts for every experiment in the paper.

## Repository structure

```text
ProSSD/
├── README.md                 # Paper introduction and repository guide
├── LICENSE                   # CC BY-NC 4.0
├── requirements.txt
├── src/
│   ├── run.py                # train_probe / build_lib / eval entry point
│   ├── models.py             # Pretrained encoder and tokenizer
│   ├── dataloader.py         # Paired and labeled JSON loaders
│   ├── method.py             # Projection, features, distributions, scoring
│   └── statistics.py         # Threshold selection and evaluation metrics
└── datasets/
    ├── README.md             # File inventory, formats, and sample counts
    ├── training_small_1800_seed_42_add.json
    ├── detectrl_domians_*     # Domain evaluation files (original spelling)
    ├── detectrl_models_*     # Model evaluation files
    └── size_datasets/        # Training-size subsets
```

## Installation

```bash
git clone https://github.com/RuoChoXio/ProSSD.git
cd ProSSD
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install a PyTorch build compatible with your CUDA environment for GPU use. The commands below explicitly select `roberta-large`; the first run downloads its tokenizer and weights unless already cached. A local model directory can also be passed to `--model-path`.

## Datasets

All released data is under [`datasets/`](datasets/README.md):

| Collection | Contents |
| --- | --- |
| Training | 1,800 human/LLM text pairs |
| Cross-domain evaluation | arXiv, writing prompts, XSum, and Yelp reviews; 2,000 labeled texts per file |
| Merged domain evaluation | 8,000 labeled texts |
| Cross-model evaluation | Claude Sonnet 4, Gemini 3 Flash Preview, GPT-5.1, and Grok 4.1; 2,000 labeled texts per file |
| Training-size subsets | 50, 100, 200, 400, 600, 800, 1,000, 1,200, 1,400, 1,600, and 1,800 pairs |

Training records contain `human_text` and `direct_prompt`. Evaluation records contain `text` and `label`, with `human` and `llm` labels. See the [dataset README](datasets/README.md) for exact filenames and loader behavior. The `domians` spelling is retained to preserve the supplied filenames.

## Run the pipeline

Run all commands from the **repository root**, in the following order. Use the same encoder, projection dimension, and output directory in all three stages. The examples use `--k 4` explicitly, since the CLI default is `32`.

### 1. Learn the supervised projection

```bash
python src/run.py train_probe --model-path roberta-large --train-data datasets/training_small_1800_seed_42_add.json --output-base experiments --k 4 --batch-size 32 --device cuda:0
```

Creates `experiments/roberta-large_k4/projection.npy`.

### 2. Build the distribution library

```bash
python src/run.py build_lib --model-path roberta-large --train-data datasets/training_small_1800_seed_42_add.json --output-base experiments --k 4 --batch-size 32 --spacy-workers 4 --device cuda:0
```

Creates `experiments/roberta-large_k4/style_lib.pkl`, including a decision threshold selected on the training data.

### 3. Evaluate

```bash
python src/run.py eval --model-path roberta-large --test-data datasets/detectrl_domians_arxiv_test.json --output-base experiments --k 4 --batch-size 32 --spacy-workers 4 --device cuda:0
```

Replace `--test-data` with another evaluation file to evaluate a different domain or model. Evaluation reuses the training threshold and writes detailed JSON reports and `summary_results.csv` under `results/` when using `--output-base experiments`. Metrics include AUROC, accuracy, precision, recall, F1, and TPR at 1% and 0.01% FPR; metric values are percentages.

For a small smoke run, pass `--limit 50` to **each stage** and choose a separate `--output-base`, such as `experiments-smoke`. This limits the number of pairs and is not a full evaluation. Reduce `--batch-size` if GPU memory is insufficient, use `--device cpu` for CPU execution, or set `--spacy-workers 1` if multiprocessing causes problems.

### Command-line arguments

| Argument | Actual CLI default | Description |
| --- | --- | --- |
| Positional subcommand | Required | `train_probe`, `build_lib`, or `eval`; not a `--task` option |
| `--model-path` | `./roberta-large` | Local model directory or Hugging Face model ID |
| `--train-data` | Required for training/library | Training JSON path |
| `--test-data` | Required for evaluation | Evaluation JSON path |
| `--output-base` | `./experiments` | Parent directory for projection and library artifacts |
| `--k` | `32` | Projection dimension; examples explicitly use `4` |
| `--batch-size` | `32` | Encoder batch size |
| `--device` | `cuda:0` | PyTorch device |
| `--spacy-workers` | `4` | spaCy processes; available for `build_lib` and `eval` |
| `--limit` | No limit | Maximum number of text pairs loaded |

The pretrained encoder is frozen, but the projection, class distributions, and threshold require labeled training data. No trained projection or distribution library is included in this release.

## License

Copyright (c) 2026 ProSSD contributors.

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE).

You may share and adapt the material for noncommercial purposes, provided you give appropriate credit, link to the license, and indicate any changes. Commercial use is not permitted under this license.

See the [official license summary](https://creativecommons.org/licenses/by-nc/4.0/) and [full license text](LICENSE). Third-party materials remain subject to their original rights and terms.

If you have any further questions, please feel free to contact the author at [xiongrc@stu.pku.edu.cn](mailto:xiongrc@stu.pku.edu.cn).
