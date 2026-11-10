# Working with LLMs

Exercises for **Week 10 of the module «Data Science»** (MSc, ZHAW School of Management and Law).
You will work with a large language model through its API: count tokens and estimate
costs, write and evaluate prompts, build a small retrieval-augmented pipeline (RAG) and
compare an API model with a local open-weight model.

The five practical exercises use real IKEA reviews from the
[Yelp Open Dataset](https://yelp.com/dataset). The practice exam uses synthetic data
and simulated model results, so it needs neither an API key nor the Yelp download.

## Getting started

1. Open <https://codespaces.new/mario-gellrich-zhaw/llm_fundamentals>.
2. GitHub asks for **`OPENAI_API_KEY`**: paste the course key you received from your
   lecturer, then click **Create codespace**. See [API key](#api-key) below.
3. Wait until VS Code opens in the browser. The first start takes several minutes while
   PyTorch, the local privacy model and Ollama are installed; later starts are fast.
4. Add the data from Moodle, see [Data](#data) below.
5. Start with [01_Explore_Yelp_Data/explore_yelp_data.ipynb](01_Explore_Yelp_Data/explore_yelp_data.ipynb).
   In [02_Chat_Loop_Tokens/chat_loop_tokens.ipynb](02_Chat_Loop_Tokens/chat_loop_tokens.ipynb),
   the first code cell should print `API key found: True`.

### API key

You don't need your own OpenAI account. You enter the course key once, when you create
the codespace. GitHub saves it as a Codespaces secret in your personal settings and makes
it available in your codespace as the environment variable `OPENAI_API_KEY`. It is never
stored in the repository.

Missed the prompt, or `API key found: False`?

1. Open <https://github.com/settings/codespaces> → **Codespaces secrets** → **New secret**.
2. Name: `OPENAI_API_KEY`, value: the course key, repository access: `llm_fundamentals`.
3. Restart the codespace: in VS Code, open the Command Palette (`F1`) and choose
   **Codespaces: Rebuild Container**, or stop and start it on <https://github.com/codespaces>.

- **Keep the key private.** Don't print it, copy it or show it in screenshots.
- **All of us share one key.** If you see `429 rate limit`, just wait: the notebooks retry
  automatically.
- **Identical cached requests reuse their answers.** `ask()`, `ask_many()` and
  `ask_decision()` cache completed, non-refused responses in `.cache/llm-v2/`.
  Changed prompts, new `run` values and `use_cache=False` incur new API calls.
  The API timing cell in `05_Local_Model` is deliberately uncached and costs money on every rerun.
  Retries are finite; persistent rate limits or exhausted quota still need attention.

Working outside Codespaces? Copy `.env.example` to `.env` and enter your own key.
`.env` is never committed.

### Data

The notebooks use a subset of the Yelp Open Dataset: all 716 McDonald's and IKEA
locations and all 1,743 IKEA reviews. The Yelp terms do not allow publishing the data,
so it is **not in this repository**. You get it on Moodle.

1. Download `llm_fundamentals_data.zip` from Moodle (Week 10).
2. In your codespace, drag the zip file from your computer into the file Explorer, onto
   the folder `data`.
3. Open [01_Explore_Yelp_Data/explore_yelp_data.ipynb](01_Explore_Yelp_Data/explore_yelp_data.ipynb)
   and run the cells up to **Unzip the data**. The notebook checks that the zip file is
   there, unpacks it into `data/`, and tells you what to do if it is missing.

The files land in `data/`. Git ignores them, so they never end up on GitHub, even if you
commit and push your work. [data/README.md](data/README.md) describes every column.

Rules for the data:

- Use it only for this course. Keep raw data local and do not share it. Only locally
  de-identified text inspected for remaining personal details may enter the course API workflow.
- Review texts can contain names of people. Don't search for or contact the authors.
- Delete the data and your codespace after the semester.

### Local installation and notebook paths

From the repository root, with Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
python -m pip install -e .
python -m spacy download en_core_web_sm
cp -n .env.example .env
```

Enter your key in `.env`, then select this environment as the notebook's Python kernel.
Codespaces runs the package and privacy-model installation automatically; rebuild an
existing Codespace or run the installation commands after updating the course.
Exercise notebooks use the installed helper's repository-relative `DATA` path and work
from the root or their own folder. The exam setup resolves either working directory too.
The optional Ollama cells require a Linux/macOS shell; the supplied Codespace is Linux.

### Anonymization before OpenAI

Every API exercise includes a local anonymization step **before** reviews enter a prompt.
Presidio recognizers and the English spaCy model detect names, locations, dates,
contact details and other identifiers, then replace matching spans with `<REDACTED>`.
Review the prepared text locally and remove remaining identifiers or identifying stories.
Automated detection can miss information and remove useful context: this is a practical
de-identification step, not a guarantee that a person cannot be identified. Use synthetic
data or local inference when sufficient de-identification is not feasible.

The course's text-only API helpers repeat redaction on input, instructions, examples and
message history before cache lookup or transmission. They stop if the privacy model is
missing; they do not fall back to sending raw text. RAG cites local passage aliases, not
Yelp IDs. Do not bypass these helpers with a separate external client, uploads or stored
conversation references. OpenAI Responses calls set `store=False`; this does not replace
checking provider retention terms. Detection uses local models, not a remote privacy API.

The original data stays in ignored `data/` files; processed labels and optional human
annotations stay in ignored `data/processed/`. Notebook outputs can contain source text;
clear outputs before sharing notebooks, and never publish the source data. Old answers
in `.cache/llm/` may contain original text: the updated helpers do not reuse that cache.
Remove that old cache locally if present. The new cache is also private and ignored.

## Exercises

| Folder | Notebook | What you do | Time |
|---|---|---|---|
| `01_Explore_Yelp_Data` | `explore_yelp_data.ipynb` | Compare McDonald's and IKEA ratings, explore the IKEA reviews: stars, time, stores, length, topics | 15 min |
| `02_Chat_Loop_Tokens` | `chat_loop_tokens.ipynb` | See why the API forgets, keep a message history, guess and then compute the cost of classifying all reviews | 10 min, together |
| `03_Classify_Reviews` | `classify_reviews.ipynb` | Create silver labels, compare a vague and a precise prompt, measure correctness, consistency and robustness, improve with meta prompting; then attack your own classifier with prompt injection and defend it | 25 + 10 min, in class |
| `04_Mini_RAG` | `mini_rag.ipynb` | Index all reviews with BM25 and E5, retrieve, answer with citations, check every claim | 25 min, in class |
| `05_Local_Model` | `local_model.ipynb` | Compare a local model with the API model on the same de-identified texts: silver-label agreement, latency and input + output API fees | about 30 min, at home |
| `06_Example_Exam_Questions` | `AP01_Large_Language_Models_Aufgaben.ipynb` | Offline 40-point practice exam; compare with `AP01_Large_Language_Models_Musterloesung.ipynb` | self-study |
| `06_Example_Exam_Questions` | `Lecture_Concept_Checks.ipynb` | Additional ungraded questions with suggested answers covering the remaining lecture objectives | self-study |

`01_Explore_Yelp_Data` needs no API key. Run practical notebooks 01–05 in order;
`05_Local_Model` uses the silver labels created in `03_Classify_Reviews`.

## Folder structure

```
llm_fundamentals/
├── .devcontainer/            Codespace configuration
│   ├── devcontainer.json
│   ├── post_create.sh
│   └── post_start.sh
├── 01_Explore_Yelp_Data/     Exercise notebooks, one folder each
│   └── explore_yelp_data.ipynb
├── 02_Chat_Loop_Tokens/
│   └── chat_loop_tokens.ipynb
├── 03_Classify_Reviews/
│   └── classify_reviews.ipynb
├── 04_Mini_RAG/
│   └── mini_rag.ipynb
├── 05_Local_Model/
│   └── local_model.ipynb
├── 06_Example_Exam_Questions/ # Offline exam, solutions, concept checks and synthetic CSVs
├── tests/                    Offline privacy, caching, parsing and chunking checks
├── data/                     Yelp subset (from Moodle, not in git)
│   ├── README.md
│   ├── businesses_mcdonalds_ikea.csv   (from Moodle)
│   ├── ikea_reviews.csv                (from Moodle)
│   ├── fewshot_examples.json           (from Moodle)
│   ├── raw/                  Full Yelp files (not in git)
│   └── processed/            Files the notebooks write (not in git)
├── scripts/                  Data preparation (for the lecturer)
│   ├── prepare_data.py
│   ├── reproduce_slide_numbers.py
│   ├── make_ap01_data.py
│   ├── build_ap01_notebooks.py
│   └── validate_notebooks.py
├── src/llm_course/           Local privacy, API calls and evaluation
│   ├── __init__.py
│   ├── client.py
│   ├── privacy.py
│   ├── evaluation.py
│   └── retrieval.py
├── .env.example
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── constraints-tested.txt
└── README.md
```

## Files

### `.devcontainer/`: Codespace configuration

| File | Purpose |
|---|---|
| `devcontainer.json` | Defines the codespace: Python 3.11 image, VS Code extensions (Python, Jupyter, Copilot, Codex and others), the `OPENAI_API_KEY` secret, port 11434 for Ollama, and the two setup scripts below. |
| `post_create.sh` | Runs once when the codespace is created: installs the Python packages from `requirements.txt`, installs the helper package `llm_course`, downloads the English privacy model, and installs Ollama for `05_Local_Model`. |
| `post_start.sh` | Runs every time the codespace starts: starts the Ollama server in the background and checks that the API key is available, without printing it. |

### Exercise notebooks

| File | Purpose |
|---|---|
| `01_Explore_Yelp_Data/explore_yelp_data.ipynb` | Unzips the data from Moodle into `data/`, then exploratory analysis with pandas and matplotlib: locations and ratings of both chains, weighted vs. unweighted averages, star distribution (J-shape), reviews per year and store, review length, topic search. No API calls. |
| `02_Chat_Loop_Tokens/chat_loop_tokens.ipynb` | Follow-up questions, message history, token counting with `tiktoken`, cost estimate for all IKEA reviews, comparison with the usage the API reports. |
| `03_Classify_Reviews/classify_reviews.ipynb` | Sentiment classification from a balanced sample of up to 200 reviews (example texts and redacted duplicates removed): silver labels from `gpt-6-luna` with high reasoning effort, vague vs. precise prompt with JSON schema, agreement with the silver reference and confusion matrix, consistency over 5 runs, robustness on irony and mixed reviews, meta prompting. Then prompt injection: two attacks against an undefended prompt, the precise prompt, fixed choices (Decisions API) and fixed choices with a defence. Saves `data/processed/silver_labels.csv`. |
| `04_Mini_RAG/mini_rag.ipynb` | Duplicate check, hybrid retrieval (BM25 + E5 + reciprocal rank fusion), token-budgeted passages and answers with local citation aliases, check of every claim, and an example of a faithful but false answer. |
| `05_Local_Model/local_model.ipynb` | Starts Ollama, downloads a small open-weight model (`qwen3:1.7b`), points the OpenAI client to `localhost:11434/v1`, classifies the same 20 de-identified reviews with an explicit final-label prompt on both models and compares silver-label agreement, seconds per review and input + output API fees. Local hardware costs are separate. |

### `data/`: Yelp subset

Only `README.md` and the two empty folders are in git. The three data files come from
Moodle (see [Data](#data)).

| File | Purpose |
|---|---|
| `README.md` | Description of every data file and column, how the subset was selected, and how to rebuild it. |
| `businesses_mcdonalds_ikea.csv` | From Moodle. All 716 McDonald's and IKEA locations from the Yelp business file, with average rating and number of reviews. |
| `ikea_reviews.csv` | From Moodle. All 1,743 reviews of the 8 IKEA locations, with stars, date and full text. |
| `fewshot_examples.json` | From Moodle. Three labelled reviews used as few-shot examples in `03_Classify_Reviews` (labels by us). They are not part of any test set. |
| `raw/` | Place for the full Yelp files, only needed to rebuild the subset. Ignored by git. |
| `processed/` | Silver labels, optional independently annotated `human_labels.csv` and complete question/review annotations `rag_relevance.csv`. Ignored by git. |

### `scripts/`: data preparation

| File | Purpose |
|---|---|
| `prepare_data.py` | Rebuilds the two CSV files in `data/` from the full Yelp Parquet files (3 GB). Only needed if the subset should be recreated. |
| `reproduce_slide_numbers.py` | Recomputes the dated, input-only slide estimates from the original local data. |
| `make_ap01_data.py` | Regenerates the synthetic exam CSVs with a fixed random seed; no model calls. |
| `build_ap01_notebooks.py` | Builds both exam versions and the concept checks; executes only the offline synthetic solution. |
| `validate_notebooks.py` | Executes practical Python cells with real redaction, fake model services and blocked network access. |

### `src/llm_course/`: shared helpers

The notebooks show analysis and evaluation directly with standard libraries. Shared
helpers keep local privacy processing and request caching consistent across exercises.

| File | Purpose |
|---|---|
| `__init__.py` | Exports repository paths, privacy preparation, API wrappers and label validation. |
| `client.py` | Text-only `ask()` and `ask_decision()` redact locally, reuse completed answers and coordinate identical concurrent requests with Linux/macOS file locks and atomic writes. Invalid cache files trigger a warning and a replacement request, which can cost money. `ask_many()` preserves input order and uses 8 workers. Loads `.env` locally and retries transient API errors. |

| `privacy.py` | Local Presidio/spaCy detection, merged-span redaction and prompt preparation. Missing dependencies stop requests. English only; inspect remaining context locally. |
| `evaluation.py` | Parse explicit final labels; distinguish refused, incomplete and malformed structured answers. |
| `retrieval.py` | Split review text within E5's tokenizer budget, preserving original spans and local citation aliases. |

### Practice exam and validation

The exam retains the reference format: eight tasks, 40 points (12 concepts, 16 error
analysis, 12 coding). Its CSVs and model outcomes are synthetic, and prices are fixed
exam assumptions rather than current quotes. Suggested answers in the separate concept
check cover generation, hallucination, annotation agreement, RAG faithfulness, agents,
deployment choices and the course's data-handling rules. Optional extensions in the
practical notebooks are outside their original time budgets.

Rebuild the synthetic data and notebooks from the repository root:

```bash
python scripts/make_ap01_data.py
python scripts/build_ap01_notebooks.py
```

The builder executes only the synthetic solution notebook, with no model calls. Editing
the builder updates both exam versions together. To run offline regression checks:

```bash
python -m unittest discover -s tests -v
```

API prices and mutable model aliases can change. For reproducible experiments, record
model/snapshot, prompt, reasoning settings, detector/model versions, evaluation data,
failures and measured usage. The workload estimates in the lecture reproduction script
are explicitly input-only and use the dated slide assumptions.

With the Moodle zip and local privacy model installed, run the practical-notebook wiring
check using fake API, embedding and Ollama services with the network blocked:

```bash
python scripts/validate_notebooks.py
```

This runs real local redaction and uses temporary data/caches. It does not test actual
hosted-model quality, E5 downloads or Ollama performance. `constraints-tested.txt`
records the directly installed dependency versions used for the offline checks on
Linux/Python 3.11; it is not a complete transitive lock.

### Files in the root folder

| File | Purpose |
|---|---|
| `.env.example` | Template for a local `.env` file with `OPENAI_API_KEY`. Only needed outside Codespaces. |
| `.gitattributes` | Keeps Linux line endings for scripts and notebooks, so they also run when edited on Windows. |
| `.gitignore` | Keeps the key (`.env`), the answer cache (`.cache/`), all Yelp data (`data/*.csv`, `data/*.json`, the Moodle zip), the lecture slides (`slides/`), files the notebooks write and Python temporary files out of git. |
| `pyproject.toml` | Describes the helper package `llm_course` so that it can be installed with `pip install -e .`. |
| `constraints-tested.txt` | Direct dependency versions used by the offline validation on Linux/Python 3.11. |
| `requirements.txt` | Python packages: `openai`, `tiktoken`, `pandas`, `scikit-learn`, `rank-bm25`, `sentence-transformers` (with CPU-only PyTorch), Jupyter and others. |
| `README.md` | This file. |

Files created while you work, all ignored by git:

| Path | Content |
|---|---|
| `.cache/llm-v2/` | Completed answers to sanitized requests and empty coordination lock files. Removing cached answers causes new paid requests. |
| `data/processed/silver_labels.csv` | Silver labels from `03_Classify_Reviews`, used in `05_Local_Model`. |

## Licence

The data comes from the [Yelp Open Dataset](https://yelp.com/dataset) and is licensed for non-commercial,
academic use only; it may not be published. That's why it is distributed on Moodle and
never stored in this repository. See [Data](#data) for the rules.
