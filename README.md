# SAFE-SPOON

## Quick Start

Choose one path:

1. Run only [server.py](server.py) with preloaded data.
2. Run the full package workflow to regenerate data/model outputs.

## 1) Run only server.py (preloaded data)

First, clone the repository. Then follow the steps below.

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

Open http://127.0.0.1:5000

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
uv run python this_needs_a_better_name.py
```

Then launch the UI:

```bash
uv run python server.py
```

Open http://127.0.0.1:5000

## Helper scripts

- [aux_scripts/data_filtering.py](aux_scripts/data_filtering.py): clean/filter dataset and write near-duplicate report.
- [aux_scripts/get_reference_corpus_data.py](aux_scripts/get_reference_corpus_data.py): build reference corpus.
- [aux_scripts/optimize_prompts.py](aux_scripts/optimize_prompts.py): compare prompt variants for topic labels.

