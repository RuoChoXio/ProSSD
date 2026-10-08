# ProSSD datasets

This directory contains the 21 JSON files distributed with the ProSSD implementation. Run the pipeline from the repository root; see the [main README](../README.md).

## File inventory

Counts below are measured from the released files. A paired record contains one human text and one generated text; a labeled record contains one text.

| File | Records | Format |
| --- | ---: | --- |
| [detectrl_domians_arxiv_test.json](detectrl_domians_arxiv_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_domians_merged.json](detectrl_domians_merged.json) | 8,000 | Labeled texts (balanced human/LLM) |
| [detectrl_domians_writing_prompt_test.json](detectrl_domians_writing_prompt_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_domians_xsum_test.json](detectrl_domians_xsum_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_domians_yelp_review_test.json](detectrl_domians_yelp_review_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_models_claude-sonnet-4-20250514_test.json](detectrl_models_claude-sonnet-4-20250514_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_models_gemini-3-flash-preview_test.json](detectrl_models_gemini-3-flash-preview_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_models_gpt-5.1_test.json](detectrl_models_gpt-5.1_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [detectrl_models_grok-4.1_test.json](detectrl_models_grok-4.1_test.json) | 2,000 | Labeled texts (balanced human/LLM) |
| [size_datasets/training_small_1800_seed_42_add_seed_42_100.json](size_datasets/training_small_1800_seed_42_add_seed_42_100.json) | 100 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_1000.json](size_datasets/training_small_1800_seed_42_add_seed_42_1000.json) | 1,000 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_1200.json](size_datasets/training_small_1800_seed_42_add_seed_42_1200.json) | 1,200 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_1400.json](size_datasets/training_small_1800_seed_42_add_seed_42_1400.json) | 1,400 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_1600.json](size_datasets/training_small_1800_seed_42_add_seed_42_1600.json) | 1,600 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_1800.json](size_datasets/training_small_1800_seed_42_add_seed_42_1800.json) | 1,800 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_200.json](size_datasets/training_small_1800_seed_42_add_seed_42_200.json) | 200 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_400.json](size_datasets/training_small_1800_seed_42_add_seed_42_400.json) | 400 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_50.json](size_datasets/training_small_1800_seed_42_add_seed_42_50.json) | 50 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_600.json](size_datasets/training_small_1800_seed_42_add_seed_42_600.json) | 600 | Human/LLM pairs |
| [size_datasets/training_small_1800_seed_42_add_seed_42_800.json](size_datasets/training_small_1800_seed_42_add_seed_42_800.json) | 800 | Human/LLM pairs |
| [training_small_1800_seed_42_add.json](training_small_1800_seed_42_add.json) | 1,800 | Human/LLM pairs |

The original `domians` spelling is retained in filenames. The `size_datasets/` directory contains training-size variants with seed 42 in their filenames; its 1,800-record file contains the same records as the main training file in a different order.

## Formats and loader behavior

All supplied files are UTF-8 JSON arrays. The loader also supports newline-delimited JSON.

### Paired training format

The fields consumed by the pipeline are `human_text` and `direct_prompt`. Other metadata includes `id`, `title`, `domain`, `len_human_text`, `len_direct_prompt`, `llm_type`, and `status`.

Illustrative record (not an actual sample):

```json
{"human_text": "A human-written passage.", "direct_prompt": "An LLM-generated passage."}
```

### Labeled evaluation format

Each record has `text`, `label`, `data_type`, and `llm_type`. The supplied labels are `human` and `llm`.

```json
{"text": "An example passage.", "label": "human", "data_type": "arxiv", "llm_type": "example"}
```

The current loader separates human and LLM records, then pairs them by order up to the smaller class count. These are loader pairs, not necessarily matched source/generation pairs. It treats labels containing `human` as human and other labels as machine text. The released evaluation files have equal class counts. `--limit` counts pairs, so `--limit 50` loads up to 50 texts from each class.

## Evaluation scope

Domain files cover arXiv, writing prompts, XSum, and Yelp reviews. Model files cover Claude Sonnet 4, Gemini 3 Flash Preview, GPT-5.1, and Grok 4.1. The merged file is an aggregate evaluation set; do not treat it and its domain files as independent additional datasets. Training-size variants are for data-efficiency experiments, not independent test splits.

## License

ProSSD's own licensable dataset contributions are licensed under [CC BY-NC 4.0](../LICENSE). Sharing and adaptation are permitted for noncommercial purposes with appropriate attribution, a license link, and an indication of changes. Underlying third-party text and datasets retain their original rights and applicable terms.
