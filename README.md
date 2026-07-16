# SAFE-SPOON

## Quick Start

Choose one path:

1. Run only [server.py](server.py) with preloaded data.
2. Run the full package workflow to regenerate data/model outputs.

## 1) Run only server.py (preloaded data)

First, clone the repository. Navigate into the root folder and follow the steps below.

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Get the data
Download demo data from [here](https://drive.google.com/file/d/1_gi02Ns2HH9O6OTEmuyuNaV61BaHVb4Z/view?usp=sharing). Unzip the data and put it in the root folder.

### Start server

```bash
uv run python server.py
```

The first run takes longer (Python compiles bytecode for all dependencies and the OS file cache is cold). Once you see the log output, open http://127.0.0.1:5000

### Download query annotations
You can annotate queries while interacting with the tool. Labels are automatically saved to [data/output/labels.csv](data/output/labels.csv) as soon as you begin annotating. You can also export annotations using the Export button at the top right of the interface.

## 2) Run full package workflow

Use this path if you want to regenerate topic models, tree structure, and UI JSON from dataset files.

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[viz]"
uv run python -m spacy download en_core_web_lg
```

### Environment variables (only needed for LLM-generated labels)

Create a [.env](.env) file at the repository root:

```dotenv
OPENAI_API_KEY=sk-...
# optional alternative variable name used by scripts
LLM_API_KEY=
```

### Input/output files used by the pipeline

- Main input: [data/high_risk_automatically_labelled_filtered_cleaned.csv](data/high_risk_automatically_labelled_filtered_cleaned.csv)
- Reference corpus: [data/reference_corpus.csv](data/reference_corpus.csv)
- Reference preprocessed cache: [data/reference_corpus_preprocessed.csv](data/reference_corpus_preprocessed.csv)
- Generated UI data: [data/output/viz_v5_data.json](data/output/viz_v5_data.json)

Required columns in the main input CSV:
- factual_analytical_label
- high_risk_label
- content

### Run pipeline

```bash
uv run safe-spoon-build
```

Useful flags: `--retrain`, `--optimize`, `--categories "Health"` (repeatable),
`--config path/to/config.yaml`. All other pipeline parameters (LDA settings,
clustering thresholds, embedding model, paths, ...) are read from
[config/config.yaml](config/config.yaml) — edit that file instead of passing flags.

Then launch the UI:

```bash
uv run python server.py
```

Open http://127.0.0.1:5000

## Command reference

All commands assume the venv from one of the Install steps above is active (or prefix with `uv run`).

### Installed CLI commands

| Command | What it does |
| --- | --- |
| `safe-spoon-build` | Runs the full pipeline (LDA training, clustering, annotation-unit labelling) and writes `data/output/viz_v5_data.json`. Flags: `--retrain`, `--optimize`, `--categories "Health"` (repeatable), `--config path/to/config.yaml`. |
| `safe-spoon-reset-annotations` | Wipes rubrics, risk profiles, LLM suggestions and unit lineage from the annotation database (`data/output/annotation.db`). Flags: `--category "Health"` (repeatable, defaults to all), `--db-path`, `--config`, `--yes` to skip the confirmation prompt. See [src/safe_spoon/annotation/reset.py](src/safe_spoon/annotation/reset.py). |

### Servers

| Command | What it does |
| --- | --- |
| `python server.py` | Starts the Flask webapp (topic viz, annotation UI) at `http://127.0.0.1:5000`. |

### Data-prep scripts (`aux_scripts/`)

Run with `python aux_scripts/<name>.py`.

| Script | What it does |
| --- | --- |
| [data_filtering.py](aux_scripts/data_filtering.py) | Cleans/filters the raw dataset (dedup, empty/name removal) and writes a near-duplicate report. |
| [get_reference_corpus_data.py](aux_scripts/get_reference_corpus_data.py) | Builds the reference corpus used to preprocess/score the LDA models. |
| [filter.sh](aux_scripts/filter.sh) | Runs `data_filtering.py` then `get_reference_corpus_data.py` back to back. |
| [optimize_prompts.py](aux_scripts/optimize_prompts.py) | Compares prompt variants for topic/annotation-unit labelling. |
| [diagnose_unit_coherence.py](aux_scripts/diagnose_unit_coherence.py) `[category]` | Compares clustering configs (linkage method, purity factor, embeddings vs. Bhattacharyya) and reports unit coherence stats. Defaults to the first `active_categories` entry in config.yaml. |
| [sample_units_for_review.py](aux_scripts/sample_units_for_review.py) `[category]` | Samples annotation units under different clustering configs for manual review. Defaults to the first `active_categories` entry in config.yaml. |
| [visualize_annotations.py](aux_scripts/visualize_annotations.py) | Builds UMAP/PCA plots of query embeddings from `data/safespoon_annotations_top5.csv`, colored by category/flag. |

## TODOs on filtering

- maybe we should exclude questions that ask for closed or one-word answers 
- I still saw quite a lot of duplicates, maybe we cna lower the threhsold for fuzzy matching again
- the personal details category is quite fuzzy, I need to rethink if we want to exclude all perosnal details or just some
- we need to think about how to deal with the NAME tags