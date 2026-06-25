.. image:: https://github.com/marlenelutz/safe-spoon/blob/main/static/jelly_logo4.png
   :align: center
   :width: 600px
===========================
SAFE-SPOON
===========================


[to be defined]

## Usage

### Steps for running the Annotation tools

### Download the data folder and copy it at the root of this repo

<insert here the link>


### Steps for running package + Annotation tool

#### Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=sk-...       # required if using llm_provider="openai"
LLM_API_KEY=                # alternative generic key (takes precedence over OPENAI_API_KEY)
```

#### Installation

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

