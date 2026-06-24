# safe-spoon

[to be defined]

## Installation

### Steps

```bash
# 1. Clone the repo
git clone <repo-url>
cd safe-spoon

# 2. Create a virtual environment and install the package in editable mode
uv venv
uv pip install -e .

# 3. Download the spaCy language model you'll use
uv run python -m spacy download en_core_web_lg # recommended
```

If you want pyLDAvis support for topic model visualisation:

```bash
uv pip install -e ".[viz]"
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=sk-...       # required if using llm_provider="openai"
LLM_API_KEY=                # alternative generic key (takes precedence over OPENAI_API_KEY)
```

# TODOS
- Remove empties, NAME_1, and queries similar with a degree of similarity larger than 90 %. DONE, annotation pending (Marlene)
- Optimize topic models (Implemented, currently optimizing)
- Add ticks for Marlene to annotate (Done, validation needed)
- Generate annotation units and add them to the platform (first version ok, further validation needed)
- Addd a second tick for demographic (done)
- Don't show to the user those that either too long or too short (implementation based on percentiles added)