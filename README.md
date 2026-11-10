# Working with LLMs

Exercises for **Week 10 of the module «Data Science»** (MSc, ZHAW School of Management and Law).
You will work with a large language model through its API: count tokens and estimate
costs, write and evaluate prompts, build a small retrieval-augmented pipeline (RAG) and
compare an API model with a local open-weight model.

The five practical exercises use real IKEA reviews from the
[Yelp Open Dataset](https://yelp.com/dataset). The practice exam uses synthetic data
and simulated model results, so it needs neither an API key nor the Yelp download.

## Getting started

1. **Fork this repository:** click **Fork** at the top right of
   <https://github.com/mario-gellrich-zhaw/llm_fundamentals>, then **Create fork**.
   You now have your own copy, `<your-username>/llm_fundamentals`, where you can save
   your work.
2. **Store the course key** as a Codespaces secret, once, before you create the
   codespace. See [API key](#api-key) below.
3. **Create the codespace from your fork**, not from this repository: open
   `https://codespaces.new/<your-username>/llm_fundamentals` (replace `<your-username>`),
   or on your fork click **Code** → **Codespaces** → **Create codespace on master**.
4. Wait until VS Code opens in the browser. The first start takes several minutes while
   PyTorch, the local privacy model and Ollama are installed; later starts are fast.
5. Add the data from Moodle, see [Data](#data) below.
6. Run [01_Explore_Yelp_Data/explore_yelp_data.ipynb](01_Explore_Yelp_Data/explore_yelp_data.ipynb)
   to the end. Its last part creates the anonymized sample of 500 reviews that notebooks
   03–05 use. In [02_Chat_Loop_Tokens/chat_loop_tokens.ipynb](02_Chat_Loop_Tokens/chat_loop_tokens.ipynb),
   the first code cell should print `API key found: True`.

### API key

You don't need your own OpenAI account. You store the course key once as a Codespaces
secret in your GitHub settings. GitHub makes it available in your codespace as the
environment variable `OPENAI_API_KEY`. It is never stored in the repository.

1. Open <https://github.com/settings/codespaces> → **Codespaces secrets** → **New secret**.
2. Name: `OPENAI_API_KEY`, value: the course key you received from your lecturer.
3. **Repository access: your fork, `<your-username>/llm_fundamentals`.** A secret only
   reaches codespaces of the repositories in this list.
4. Click **Add secret**.

Depending on how you create the codespace, GitHub may also ask for `OPENAI_API_KEY` on
the creation page. If the secret is already stored, you can ignore that field.

`API key found: False`?

- Check the secret on <https://github.com/settings/codespaces>: is the name exactly
  `OPENAI_API_KEY`, and is **your fork** in the repository access list?
- Was the codespace created from your fork? Its address bar and the bottom-left corner
  of VS Code show the repository.
- Restart the codespace after you add or change the secret: open the Command Palette
  (`F1`) and choose **Codespaces: Stop Current Codespace**, then open it again on
  <https://github.com/codespaces>. Secrets are only loaded when a codespace starts.

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

### Codespaces costs

When you create a codespace, GitHub shows *«Codespace usage for this repository is paid
for by <your-username>»*. That is expected: codespaces of personal accounts are always
paid by the person who creates them, from that person's monthly free quota. Check the
current quota and your usage on <https://github.com/settings/billing>.

- **The free quota is enough for this course** if you use the default 2-core machine.
  The quota counts core-hours: a 4-core machine uses it twice as fast. Switch back to
  2 cores after the optional speed test in `05_Local_Model`.
- **The spending limit is USD 0 by default.** When the quota is used up, codespaces stop;
  nothing is charged unless you raise the limit.
- **Stopped codespaces still use storage.** PyTorch, the embedding model and the Ollama
  model make this codespace a few GB large. Delete it on <https://github.com/codespaces>
  when you no longer need it; push your work to your fork first.
- **Idle codespaces stop after 30 minutes** by default, so a forgotten browser tab does
  not use up your quota.

Your fork is public, like this repository. Push only your code and notebooks: the data,
`.env` and the answer cache are ignored by Git and never leave the codespace.

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

### Anonymization before the API

Reviews can contain names and other personal details. **No personal data may be sent to
an external API.** The course handles this in two steps:

1. **Notebook 01 prepares a sample.** Its last part draws a random sample of 500 reviews,
   removes personal details with `anonymize()` and saves the result as
   `data/processed/reviews_sample.csv`. You compare texts before and after and can correct
   them by hand. `anonymize()` runs in your codespace (libraries Presidio and spaCy) and
   replaces details that identify a person with `<REDACTED>`: names of people, e-mail
   addresses, phone numbers and account or ID numbers. Company names, places and dates
   stay, because the model needs them to understand a review. No API key is needed.
2. **The helpers check again.** `ask()`, `ask_many()` and `ask_decision()` anonymize every
   text once more right before it is sent. If the privacy model is missing, they stop
   instead of sending the original text.

Notebooks 03–05 only read the prepared sample. They stop with a clear message if you have
not run notebook 01 (or, for notebook 05, notebook 03) yet. Notebook 02 counts the tokens
of all reviews on your computer and sends none of them.

Automatic anonymization makes mistakes: it can miss a name and it can remove harmless
words. It reduces the risk; it does not guarantee that nobody can be identified. Always
call the API through the course helpers, not through your own client. The sample, the
answer cache and notebook outputs can contain review text: keep them private, and clear
the outputs before you share a notebook.

## Exercises

| Folder | Notebook | What you do | Time | First run |
|---|---|---|---|---|
| `01_Explore_Yelp_Data` | `explore_yelp_data.ipynb` | Compare McDonald's and IKEA ratings, explore the IKEA reviews, prepare an anonymized sample of 500 reviews | 15 min | under 1 min |
| `02_Chat_Loop_Tokens` | `chat_loop_tokens.ipynb` | See why the API forgets, keep a message history, count tokens, compute the cost of reading all reviews, see what reasoning costs | 10 min, together | under 1 min |
| `03_Classify_Reviews` | `classify_reviews.ipynb` | Label 100 reviews, compare a vague and a precise prompt, measure correctness, consistency and robustness, improve the prompt; then attack your own classifier with prompt injection and defend it | 25 + 10 min, in class | about 3 min |
| `04_Mini_RAG` | `mini_rag.ipynb` | Search 500 reviews by keywords (BM25) and by meaning (E5), let the model answer with sources, check the answer | 25 min, in class | about 2 min |
| `05_Local_Model` | `local_model.ipynb` | Run a small local model with Ollama and compare accuracy, speed and cost with the API model on 20 reviews | 15 min, at home | about 2 min + model download |
| `06_Example_Exam_Questions` | `AP01_Large_Language_Models_Aufgaben.ipynb` | Offline 40-point practice exam; compare with `AP01_Large_Language_Models_Musterloesung.ipynb` | self-study | seconds |
| `06_Example_Exam_Questions` | `Lecture_Concept_Fragen_und_Antworten.ipynb` | Additional ungraded questions with suggested answers on the remaining lecture topics | self-study | – |

"First run" is the time to run all cells once in a 2-core codespace. Later runs are faster,
because answers of the model are saved in `.cache/`.

**How the notebooks keep it simple and fast**

- Every step starts with **one example** (one review, one request) before it is applied to many.
- The API notebooks use **small samples**: 500 anonymized reviews in total, 100 for
  classification, 20 for the model comparison.
- Normal requests use **low reasoning effort**; only the silver labels use high effort.
  The local model answers without thinking by default.

`01_Explore_Yelp_Data` needs no API key. Run the notebooks 01–05 in order: 03 and 04 use
the sample created in 01, and `05_Local_Model` uses the silver labels created in
`03_Classify_Reviews`.

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
├── tests/                    Offline checks: privacy, caching, parsing, notebook cells
├── data/                     Yelp subset (from Moodle, not in git)
│   ├── README.md
│   ├── businesses_mcdonalds_ikea.csv   (from Moodle)
│   ├── ikea_reviews.csv                (from Moodle)
│   ├── fewshot_examples.json           (from Moodle)
│   ├── raw/                  Full Yelp files (not in git)
│   └── processed/            Files the notebooks write (not in git)
├── scripts/                  Data preparation and checks (for the lecturer), Ollama start
│   ├── prepare_data.py
│   ├── reproduce_slide_numbers.py
│   ├── make_ap01_data.py
│   ├── build_ap01_notebooks.py
│   ├── start_ollama.sh
│   └── validate_notebooks.py
├── src/llm_course/           Local privacy, API calls and evaluation
│   ├── __init__.py
│   ├── client.py
│   ├── privacy.py
│   └── evaluation.py
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
| `01_Explore_Yelp_Data/explore_yelp_data.ipynb` | Unzips the data from Moodle into `data/`. Exploratory analysis with pandas and matplotlib: locations and ratings of both chains, weighted vs. unweighted averages, star distribution (J-shape), reviews per year and store, review length, topic search. The last part draws a random sample of 500 unique reviews, anonymizes it locally, shows texts before and after, and saves `data/processed/reviews_sample.csv`. No API calls. |
| `02_Chat_Loop_Tokens/chat_loop_tokens.ipynb` | Follow-up question without and with a message history, the same prompt three times, tokens with `tiktoken`, cost of reading all IKEA reviews, context window, usage reported by the API, low vs. high reasoning effort. About 12 small API calls. |
| `03_Classify_Reviews/classify_reviews.ipynb` | Sentiment of 100 reviews (20 per star rating): one review as free text and with a JSON schema, silver labels from `gpt-6-luna` with high reasoning effort, development and test set, vague vs. precise prompt with few-shot examples, accuracy and confusion matrix, consistency over 5 runs, hard cases, meta prompting. Then prompt injection: two attacks against four classifiers, including fixed choices (Decisions API). Optional: Cohen's kappa for two annotators, a schema with aspects and evidence. Saves `data/processed/silver_labels.csv`. |
| `04_Mini_RAG/mini_rag.ipynb` | Keyword search (BM25) and search by meaning (E5) over the 500 sample reviews, both combined with reciprocal rank fusion, answers that cite review IDs (`R001`–`R500`), check of the cited reviews, and a made-up review that leads to a faithful but false answer. |
| `05_Local_Model/local_model.ipynb` | Starts Ollama and downloads a small open-weight model (`qwen3:1.7b`), points the OpenAI library to `localhost:11434/v1`, shows what thinking costs in time, classifies 20 reviews with the local model and with `gpt-6-luna`, and compares accuracy against the silver labels, seconds per review and API cost. |

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
| `processed/` | Files the notebooks write: the anonymized sample from notebook 01 and the silver labels from notebook 03. Ignored by git. |

### `scripts/`: data preparation

| File | Purpose |
|---|---|
| `prepare_data.py` | Rebuilds the two CSV files in `data/` from the full Yelp Parquet files (3 GB). Only needed if the subset should be recreated. |
| `reproduce_slide_numbers.py` | Recomputes the dated, input-only slide estimates from the original local data. |
| `make_ap01_data.py` | Regenerates the synthetic exam CSVs with a fixed random seed; no model calls. |
| `build_ap01_notebooks.py` | Builds both exam versions and the concept checks; executes only the offline synthetic solution. |
| `start_ollama.sh` | Used by `05_Local_Model`: installs Ollama if it is missing, starts the server and downloads the model. |
| `validate_notebooks.py` | Executes practical Python cells with real redaction, fake model services and blocked network access. |

### `src/llm_course/`: shared helpers

The notebooks show every analysis step directly with standard libraries (pandas,
scikit-learn, tiktoken, rank-bm25, sentence-transformers). Only two things are hidden in
helpers, because they must work the same way in every notebook: removing personal details
and calling the API with a cache.

| Function | What it does |
|---|---|
| `ask(...)` | Sends one request, like `client.responses.create(...)` on the slides. Anonymizes the text first and saves the answer in `.cache/`: the same request is not sent twice. `run=1, 2, ...` asks for a new answer, `use_cache=False` always sends the request. |
| `ask_many(inputs, ...)` | `ask()` for a list of inputs, 8 at a time, with a progress bar. |
| `ask_decision(...)` | Fixed choices (Decisions API) with the same anonymization and cache. |
| `anonymize(text)` | Replaces names of people, e-mail addresses, phone numbers and account or ID numbers with `<REDACTED>`. Runs locally. |
| `response_label(response)` | Reads the label from a JSON answer; returns `'unparsed'`, `'refused'` or `'incomplete'` if there is no valid label. |
| `parse_label(text)` | Accepts a free-text answer only if it is exactly one label; otherwise `'unparsed'`. |
| `DATA`, `ROOT` | Paths of the folder `data` and of the repository. |

| File | Purpose |
|---|---|
| `__init__.py` | Makes the functions above available with `from llm_course import ...`. |
| `client.py` | `ask()`, `ask_many()` and `ask_decision()`: anonymization before sending, answer cache on disk (safe when several notebooks run at once), automatic retries on rate limits, key from `.env` outside Codespaces. |
| `privacy.py` | `anonymize()` with Presidio and the English spaCy model, and the check that prepares every request. Stops the request if the privacy model is missing. English only. |
| `evaluation.py` | `parse_label()` and `response_label()`. |

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
| `data/processed/reviews_sample.csv` | Anonymized sample of 500 reviews from notebook 01, used in notebooks 03 and 04. |
| `data/processed/silver_labels.csv` | 100 reviews with silver labels from `03_Classify_Reviews`, used in `05_Local_Model`. |

## Licence

The data comes from the [Yelp Open Dataset](https://yelp.com/dataset) and is licensed for non-commercial,
academic use only; it may not be published. That's why it is distributed on Moodle and
never stored in this repository. See [Data](#data) for the rules.
