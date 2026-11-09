# Working with LLMs

Exercises for **Week 10 of the module «Data Science»** (MSc, ZHAW School of Management and Law).
You will work with a large language model through its API: count tokens and estimate
costs, write and evaluate prompts, build a small retrieval-augmented pipeline (RAG) and
compare an API model with a local open-weight model.

All exercises use one data set: real customer reviews of IKEA from the
[Yelp Open Dataset](https://yelp.com/dataset).

## Getting started

1. Open <https://codespaces.new/mario-gellrich-zhaw/llm_fundamentals>.
2. GitHub asks for **`OPENAI_API_KEY`**: paste the course key you received from your
   lecturer, then click **Create codespace**. See [API key](#api-key) below.
3. Wait until VS Code opens in the browser. The first start takes several minutes while
   PyTorch and Ollama are installed; later starts are fast.
4. Add the data from Moodle, see [Data](#data) below.
5. Start with [01_Explore_Yelp_Data/explore_yelp_data.ipynb](01_Explore_Yelp_Data/explore_yelp_data.ipynb).
   In [02_Chat_Loop_Tokens/chat_loop_tokens.ipynb](02_Chat_Loop_Tokens/chat_loop_tokens.ipynb),
   the first code cell should print `API key found.`

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
- **Running a cell again is free.** Answers are saved in `.cache/`; the same request is
  not sent twice.

Working outside Codespaces? Copy `.env.example` to `.env` and enter your own key.
`.env` is never committed.

### Data

The notebooks use a subset of the Yelp Open Dataset: all 716 McDonald's and IKEA
locations and all 1,743 IKEA reviews. The Yelp terms do not allow publishing the data,
so it is **not in this repository**. You get it on Moodle.

1. Download `llm_fundamentals_data.zip` from Moodle (Week 10).
2. In your codespace, drag the zip file from your computer into the file Explorer, onto
   the folder `data`.
3. In the terminal, run:

   ```bash
   python -m zipfile -e data/llm_fundamentals_data.zip data
   ```

The files land in `data/`. Git ignores them, so they never end up on GitHub, even if you
commit and push your work. [data/README.md](data/README.md) describes every column.

Rules for the data:

- Use it only for this course. Don't upload it anywhere else, and don't share it.
- Review texts can contain names of people. Don't search for or contact the authors.
- Delete the data and your codespace after the semester.

## Exercises

| Folder | Notebook | What you do | Time |
|---|---|---|---|
| `01_Explore_Yelp_Data` | `explore_yelp_data.ipynb` | Compare McDonald's and IKEA ratings, explore the IKEA reviews: stars, time, stores, length, topics | 15 min |
| `02_Chat_Loop_Tokens` | `chat_loop_tokens.ipynb` | See why the API forgets, keep a message history, guess and then compute the cost of classifying all reviews | 10 min, together |
| `03_Classify_Reviews` | `classify_reviews.ipynb` | Create silver labels, compare a vague and a precise prompt, measure correctness, consistency and robustness, improve with meta prompting; then attack your own classifier with prompt injection and defend it | 25 + 10 min, in class |
| `04_Mini_RAG` | `mini_rag.ipynb` | Index all reviews with BM25 and E5, retrieve, answer with citations, check every claim | 25 min, in class |
| `05_Local_Model` | `local_model.ipynb` | Run a local model with Ollama and compare accuracy, speed and cost with the API model | about 30 min, at home |

`01_Explore_Yelp_Data` needs no API key. Run the notebooks in this order:
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
├── data/                     Yelp subset (from Moodle, not in git)
│   ├── README.md
│   ├── businesses_mcdonalds_ikea.csv   (from Moodle)
│   ├── ikea_reviews.csv                (from Moodle)
│   ├── fewshot_examples.json           (from Moodle)
│   ├── raw/                  Full Yelp files (not in git)
│   └── processed/            Files the notebooks write (not in git)
├── scripts/                  Data preparation (for the lecturer)
│   ├── prepare_data.py
│   └── reproduce_slide_numbers.py
├── src/llm_course/           Two helpers for API calls
│   ├── __init__.py
│   └── client.py
├── .env.example
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Files

### `.devcontainer/`: Codespace configuration

| File | Purpose |
|---|---|
| `devcontainer.json` | Defines the codespace: Python 3.11 image, VS Code extensions (Python, Jupyter, Copilot, Codex and others), the `OPENAI_API_KEY` secret, port 11434 for Ollama, and the two setup scripts below. |
| `post_create.sh` | Runs once when the codespace is created: installs the Python packages from `requirements.txt`, installs the helper package `llm_course`, and installs Ollama for `05_Local_Model`. |
| `post_start.sh` | Runs every time the codespace starts: starts the Ollama server in the background and checks that the API key is available, without printing it. |

### Exercise notebooks

| File | Purpose |
|---|---|
| `01_Explore_Yelp_Data/explore_yelp_data.ipynb` | Exploratory analysis with pandas and matplotlib: locations and ratings of both chains, weighted vs. unweighted averages, star distribution (J-shape), reviews per year and store, review length, topic search. No API calls. |
| `02_Chat_Loop_Tokens/chat_loop_tokens.ipynb` | Follow-up questions, message history, token counting with `tiktoken`, cost estimate for all IKEA reviews, comparison with the usage the API reports. |
| `03_Classify_Reviews/classify_reviews.ipynb` | Sentiment classification of 200 reviews: silver labels from `gpt-6-luna` with high reasoning effort, vague vs. precise prompt with JSON schema, accuracy and confusion matrix, consistency over 5 runs, robustness on irony and mixed reviews, meta prompting. Then prompt injection: two attacks against an undefended prompt, the precise prompt, fixed choices (Decisions API) and fixed choices with a defence. Saves `data/processed/silver_labels.csv`. |
| `04_Mini_RAG/mini_rag.ipynb` | Duplicate check, hybrid retrieval (BM25 + E5 + reciprocal rank fusion), answers with cited review IDs, check of every claim, and an example of a faithful but false answer. |
| `05_Local_Model/local_model.ipynb` | Starts Ollama, downloads a small open-weight model (`qwen3:1.7b`), points the OpenAI client to `localhost:11434/v1`, classifies 20 reviews with the same one-word prompt on both models and compares accuracy, seconds per review and API cost. |

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
| `processed/` | Files the notebooks write, for example the silver labels. Ignored by git. |

### `scripts/`: data preparation

| File | Purpose |
|---|---|
| `prepare_data.py` | Rebuilds the two CSV files in `data/` from the full Yelp Parquet files (3 GB). Only needed if the subset should be recreated. |
| `reproduce_slide_numbers.py` | Recomputes the numbers shown in the lecture (average ratings, star distribution, token counts, costs, duplicate reviews) from the CSV files in `data/`. |

### `src/llm_course/`: two helpers for API calls

Everything else in the notebooks uses standard libraries directly: pandas, scikit-learn,
tiktoken, rank-bm25, sentence-transformers and the OpenAI library. Only API calls go
through two small functions, because all students share one key.

| File | Purpose |
|---|---|
| `__init__.py` | Makes both helpers available with `from llm_course import ask, ask_many`. |
| `client.py` | `ask()` works exactly like `client.responses.create()` and returns the same response object, but saves every answer on disk, so running a cell again costs nothing. `ask_many()` calls `ask()` for many inputs, 8 at a time. The OpenAI client retries automatically on rate limits. Also loads the key from `.env` when you work outside Codespaces. |

### Files in the root folder

| File | Purpose |
|---|---|
| `.env.example` | Template for a local `.env` file with `OPENAI_API_KEY`. Only needed outside Codespaces. |
| `.gitattributes` | Keeps Linux line endings for scripts and notebooks, so they also run when edited on Windows. |
| `.gitignore` | Keeps the key (`.env`), the answer cache (`.cache/`), all Yelp data (`data/*.csv`, `data/*.json`, the Moodle zip), the lecture slides (`slides/`), files the notebooks write and Python temporary files out of git. |
| `pyproject.toml` | Describes the helper package `llm_course` so that it can be installed with `pip install -e .`. |
| `requirements.txt` | Python packages: `openai`, `tiktoken`, `pandas`, `scikit-learn`, `rank-bm25`, `sentence-transformers` (with CPU-only PyTorch), Jupyter and others. |
| `README.md` | This file. |

Files created while you work, all ignored by git:

| Path | Content |
|---|---|
| `.cache/llm/` | Saved API answers. Delete the folder to send all requests again. |
| `data/processed/silver_labels.csv` | Silver labels from `03_Classify_Reviews`, used in `05_Local_Model`. |

## Licence

The data comes from the [Yelp Open Dataset](https://yelp.com/dataset) and is licensed for non-commercial,
academic use only; it may not be published. That's why it is distributed on Moodle and
never stored in this repository. See [Data](#data) for the rules.
