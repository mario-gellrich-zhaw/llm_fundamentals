# Working with LLMs

Exercises for **Week 10 of the module «Data Science»** (MSc, ZHAW School of Management and Law).
You will work with a large language model through its API: count tokens and estimate
costs, write and evaluate prompts, build a small retrieval-augmented pipeline (RAG) and
compare an API model with a local open-weight model.

The data-preparation notebook and four LLM exercises use real IKEA and McDonald's reviews from the
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
   to the end. Its last part creates the anonymized sample of 500 reviews (250 per chain) that notebooks
   03–05 use. In [02_Chat_Loop_Tokens/chat_loop_tokens.ipynb](02_Chat_Loop_Tokens/chat_loop_tokens.ipynb),
   run the first question to check that the API connection works.

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

An API request fails because the key is missing or invalid?

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
- **Every run of a cell sends its requests again.** There is no answer cache: rerunning
  a notebook incurs new API calls, and answers can differ between runs.
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
`.env` is ignored by Git and never leaves the codespace.

### Data

The notebooks use a subset of the Yelp Open Dataset: all 718 McDonald's and IKEA
locations, all 1,743 IKEA reviews and all 18,280 McDonald's reviews. The Yelp terms do not allow publishing the data,
so it is **not in this repository**. You get it on Moodle.

1. Download `llm_fundamentals_data.zip` from Moodle (Week 10).
2. In your codespace, drag the zip file from your computer into the file Explorer, onto
   the folder `data`.
3. Open [01_Explore_Yelp_Data/explore_yelp_data.ipynb](01_Explore_Yelp_Data/explore_yelp_data.ipynb)
   and run the cells up to **Unzip the data**. The notebook checks that the zip file is
   there, unpacks it into `data/`, and tells you what to do if it is missing.

The files land in `data/`. Git ignores them, so they never end up on GitHub, even if you
commit and push your work. [data/README.md](data/README.md) describes every column.

Updating an existing course copy? Use the updated Moodle zip containing
`mcdonalds_reviews.csv`, then rerun notebooks 01 and 03 in full. This regenerates the
anonymized sample and silver labels with both chains for notebooks 04 and 05.

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
python -m spacy download en_core_web_sm
cp -n .env.example .env
```

Enter your key in `.env`, then select this environment as the notebook's Python kernel.
Codespaces runs the package and privacy-model installation automatically; rebuild an
existing Codespace or run the installation commands after updating the course.
Exercise notebooks read the data from `../data`, so run them from their own folder (the
default in VS Code). The exam setup works from its own folder or from the repository root.
The optional Ollama cells require a Linux/macOS shell; the supplied Codespace is Linux.

### Anonymization before the API

Reviews can contain names and other personal details. **No personal data may be sent to
an external API.** The course handles this in two steps:

1. **Notebook 01 prepares a sample.** Its last part draws a random sample of 500 reviews (250 per chain),
   removes personal details and saves the result as
   `data/processed/reviews_sample.csv`. You compare texts before and after and can correct
   them by hand. The notebook defines a short function `anonymize()` with the libraries
   Presidio and spaCy. It runs in your codespace and replaces details that identify a
   person with a placeholder such as `<PERSON>`: names of people, e-mail addresses, phone
   numbers and account or ID numbers. Company names, places and dates stay, because the
   model needs them to understand a review. No API key is needed.
2. **Notebooks 03–05 send only this sample.** They call the OpenAI library directly
   (`client.responses.create(...)`) and do not anonymize again, so never send texts from
   the original CSV files. Requests with review texts use `store=False`, which asks the
   provider not to keep them.

Notebooks 03–05 only read the prepared sample. They stop with a clear message if you have
not run notebook 01 (or, for notebook 05, notebook 03) yet. Notebook 02 counts the tokens
of all reviews on your computer and sends none of them.

Automatic anonymization makes mistakes: it can miss a name and it can remove harmless
words. It reduces the risk; it does not guarantee that nobody can be identified. Always
send only texts from the prepared sample. The sample and notebook outputs can contain
review text: keep them private, and clear the outputs before you share a notebook.

## Exercises

| Folder | Notebook | What you do | Time | First run |
|---|---|---|---|---|
| `01_Explore_Yelp_Data` | `explore_yelp_data.ipynb` | Compare chain ratings, compare both review distributions, inspect examples and prepare an anonymized sample of 250 reviews per chain | 10–15 min, plus self-check | under 1 min |
| `02_Chat_Loop_Tokens` | `chat_loop_tokens.ipynb` | See why the API forgets, keep a message history, count tokens, compare the cost of reading both complete review datasets, see what reasoning costs | 10 min, together | under 1 min |
| `03_Classify_Reviews` | `classify_reviews.ipynb` | Label 20 reviews (10 per chain), compare two predefined prompts, check correctness, consistency and hard cases, then try injection on one review with and without a defence | 20 + 5 min, plus 5 min self-check | depends on API latency |
| `04_Mini_RAG` | `mini_rag.ipynb` | Search 250 McDonald's reviews by keywords (BM25) and by meaning (E5), let the model answer with sources, check one answer and examine one misleading source | 20 min, plus self-check | about 2 min |
| `05_Local_Model` | `local_model.ipynb` | Run a small local model with Ollama and compare accuracy, speed and cost with the API model on 10 reviews (5 per chain) | 15 min, at home | about 2 min + model download |
| `06_Example_Exam_Questions` | `AP01_Large_Language_Models_Aufgaben.ipynb` | Offline 40-point practice exam; compare with `AP01_Large_Language_Models_Musterloesung.ipynb` | self-study | seconds |
| `06_Example_Exam_Questions` | `Lecture_Concept_Fragen_und_Antworten.ipynb` | Additional ungraded questions with suggested answers on the remaining lecture topics | self-study | – |

"First run" is the time to run all cells once in a 2-core codespace. Notebooks that call
the API take about as long on every run, because answers are not saved.

**How the notebooks keep it simple and fast**

- Every step starts with **one example** (one review, one request) before it is applied to many.
- The API notebooks use **small samples**: 500 anonymized reviews in total (250 per chain), 20 for
  classification, 10 for the model comparison.
- Classification requests use **low reasoning effort**. Silver labels are prepared once
  by the lecturer and included in the Moodle data zip.
  The local model answers without thinking by default.

`01_Explore_Yelp_Data` needs no API key. Run the notebooks 01–05 in order: 03 and 04 use
the sample created in 01, and `05_Local_Model` uses the prepared silver labels saved in
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
├── tests/                    Offline checks: notebook cells and the practice exam
├── data/                     Yelp subset (from Moodle, not in git)
│   ├── README.md
│   ├── businesses_mcdonalds_ikea.csv   (from Moodle)
│   ├── ikea_reviews.csv                (from Moodle)
│   ├── mcdonalds_reviews.csv            (from Moodle)
│   ├── fewshot_examples.json           (from Moodle)
│   ├── raw/                  Full Yelp files (not in git)
│   └── processed/            Files the notebooks write (not in git)
├── scripts/                  Data preparation and checks (for the lecturer)
│   ├── prepare_data.py
│   ├── reproduce_slide_numbers.py
│   ├── make_ap01_data.py
│   ├── build_ap01_notebooks.py
│   └── validate_notebooks.py
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
| `post_create.sh` | Runs once when the codespace is created: installs the Python packages from `requirements.txt`, downloads the English privacy model, and installs Ollama for `05_Local_Model`. |
| `post_start.sh` | Runs every time the codespace starts: starts the Ollama server in the background and checks that the API key is available, without printing it. |

### Exercise notebooks

| File | Purpose |
|---|---|
| `01_Explore_Yelp_Data/explore_yelp_data.ipynb` | Unzips the data from Moodle into `data/`. Exploratory analysis with pandas and matplotlib: ratings of both chains, weighted vs. unweighted averages, review star distribution and sampling bias, three example texts, and a simple topic search. The last part draws a random sample of 500 unique reviews, anonymizes it locally, shows texts before and after, and saves `data/processed/reviews_sample.csv`. No API calls. |
| `02_Chat_Loop_Tokens/chat_loop_tokens.ipynb` | Follow-up question without and with a message history, the same prompt three times, tokens with `tiktoken`, cost of reading all IKEA and McDonald's reviews, context window, usage reported by the API, low vs. high reasoning effort. 10 small API calls; usage and low/high reasoning are compared in one experiment. |
| `03_Classify_Reviews/classify_reviews.ipynb` | Four procedural steps with simple loops on 20 reviews (10 per chain; two per star rating and chain): free text vs. JSON schema; two predefined prompts with prepared silver labels from the Moodle data zip and separate few-shot examples; accuracy, three runs on three reviews and inspection of hard cases; one made-up negative review with two injections, with and without a defence. No meta-prompting workflow or advanced schema code. Development/test separation is explained for subsequent prompt tuning. Saves `data/processed/silver_labels.csv` for notebook 05. |
| `04_Mini_RAG/mini_rag.ipynb` | Keyword search (BM25) and search by meaning (E5) over 250 McDonald's sample reviews (switchable to IKEA), combined step by step with rank fusion. Runs one complete RAG question, checks its citations and claims, then uses one made-up source to distinguish faithfulness from truth. Two generation calls in total. |
| `05_Local_Model/local_model.ipynb` | Starts Ollama and downloads a small open-weight model (`qwen3:1.7b`), points the OpenAI library to `localhost:11434/v1`, classifies the same 10 reviews with the local model and with `gpt-6-luna`, and compares accuracy against the silver labels, seconds per review and API cost. |

### `data/`: Yelp subset

Only `README.md` and the empty folder placeholders are in git. The data files come from
Moodle (see [Data](#data)).

| File | Purpose |
|---|---|
| `README.md` | Description of every data file and column, how the subset was selected, and how to rebuild it. |
| `businesses_mcdonalds_ikea.csv` | From Moodle. All 718 McDonald's and IKEA locations from the Yelp business file, with average rating and number of reviews. |
| `ikea_reviews.csv` | From Moodle. All 1,743 reviews of the 8 IKEA locations, with stars, date and full text. |
| `mcdonalds_reviews.csv` | From Moodle. All 18,280 reviews of the 710 McDonald's locations, with stars, date and full text. |
| `fewshot_examples.json` | From Moodle. Three labelled reviews used as few-shot examples in `03_Classify_Reviews` (labels by us). They are not part of any test set. |
| `raw/` | Place for the full Yelp files, only needed to rebuild the subset. Ignored by git. |
| `processed/` | Files the notebooks write: the anonymized sample from notebook 01 and the silver labels from notebook 03. Ignored by git. |

### `scripts/`: data preparation

| File | Purpose |
|---|---|
| `prepare_data.py` | Rebuilds the three CSV files in `data/` from the full Yelp Parquet files (3 GB). Only needed if the subset should be recreated. |
| `reproduce_slide_numbers.py` | Recomputes the dated, input-only slide estimates from the original local data. |
| `make_ap01_data.py` | Regenerates the synthetic exam CSVs with a fixed random seed; no model calls. |
| `build_ap01_notebooks.py` | Builds both exam versions and the concept checks; executes only the offline synthetic solution. |
| `validate_notebooks.py` | Executes practical Python cells with real redaction, fake model services and blocked network access. |

### No hidden helpers

The notebooks show every step directly with existing libraries: pandas, scikit-learn,
tiktoken, rank-bm25, sentence-transformers, Presidio and the OpenAI library. There is no
course package to install.

| Task | Where and how |
|---|---|
| Call the model | `client = OpenAI(max_retries=8)` and `client.responses.create(...)` in notebooks 02–05. The client retries automatically on rate limits. |
| API key | Codespaces secret `OPENAI_API_KEY`; outside Codespaces `load_dotenv("../.env")` reads it from `.env`. |
| Remove personal details | Function `anonymize()` defined in notebook 01 with Presidio's `AnalyzerEngine` and `AnonymizerEngine`. English only. |
| Read a label | `json.loads(r.output_text)["sentiment"]` for JSON answers (notebook 03); cleaned free text for the comparison in notebook 05. |
| Paths | `DATA = Path("../data")`, relative to the folder of the notebook. |

### Practice exam and validation

The exam retains the reference format: eight tasks, 40 points (12 concepts, 16 error
analysis, 12 coding). All tasks and sub-tasks are independent: each faulty
snippet in part B has its own inputs and exactly one error, each coding task in part C
has one TODO, and values needed for written answers are given in the task text. Its CSVs and model outcomes are synthetic, and prices are fixed
exam assumptions rather than current quotes. Suggested answers in the separate concept
check cover generation, hallucination, annotation agreement, RAG faithfulness, agents,
deployment choices and the course's data-handling rules. Optional extensions in the
practical notebooks are outside their original time budgets.

Rebuild the synthetic data and notebooks from the repository root (notebook
generation uses Black from the optional development tools):

```bash
python -m pip install "black>=25,<27" "pylint>=3.3,<5"
python scripts/make_ap01_data.py
python scripts/build_ap01_notebooks.py
```

The builder executes only the synthetic solution notebook, with no model calls.
Shared assumptions in `scripts/exam_settings.py` give both models 185 prompt tokens
per review. Simulated output totals include reasoning tokens for both models.
`scripts/exam_metrics.py` derives the numbers in tasks and solutions from the CSVs.
The unused `llm_answers.csv` is no longer generated. Cell IDs are deterministic,
and the footer omits timestamps so repeated builds do not create artificial diffs. Editing
the builder updates both exam versions together. To run offline regression checks:

```bash
python -m unittest discover -s tests -v
```

Formatting uses an 80-character line limit. Install the optional development tools
and run the checks from the repository root:

```bash
python -m pip install "black>=25,<27" "pylint>=3.3,<5"
python -m black --check scripts tests
python scripts/lint_code.py
```

The pylint check includes all Python modules and temporary exports of the notebook
code cells. Notebook-specific exceptions cover displayed expressions, exercise
TODOs, shared variable names and imports spread across cells. Code is checked
without executing model requests or exporting notebook outputs.


API prices and mutable model aliases can change. For reproducible experiments, record
model/snapshot, prompt, reasoning settings, detector/model versions, evaluation data,
failures and measured usage. The workload estimates in the lecture reproduction script
are explicitly input-only and use the dated slide assumptions.

With the Moodle zip and local privacy model installed, run the practical-notebook wiring
check using fake API, embedding and Ollama services with the network blocked:

```bash
python scripts/validate_notebooks.py
```

This runs real local redaction and uses temporary data. It does not test actual
hosted-model quality, E5 downloads or Ollama performance. `constraints-tested.txt`
records the directly installed dependency versions used for the offline checks on
Linux/Python 3.11; it is not a complete transitive lock.

### Files in the root folder

| File | Purpose |
|---|---|
| `.env.example` | Template for a local `.env` file with `OPENAI_API_KEY`. Only needed outside Codespaces. |
| `.gitattributes` | Keeps Linux line endings for scripts and notebooks, so they also run when edited on Windows. |
| `.gitignore` | Keeps the key (`.env`), the local copy of the embedding model (`models/`), all Yelp data (`data/*.csv`, `data/*.json`, the Moodle zip), the lecture slides (`slides/`), files the notebooks write and Python temporary files out of git. |
| `pyproject.toml` | Settings of the code checks (Black and pylint). |
| `constraints-tested.txt` | Direct dependency versions used by the offline validation on Linux/Python 3.11. |
| `requirements.txt` | Python packages: `openai`, `tiktoken`, `pandas`, `scikit-learn`, `rank-bm25`, `sentence-transformers` (with CPU-only PyTorch), Jupyter and others. |
| `README.md` | This file. |

Files created while you work, all ignored by git:

| Path | Content |
|---|---|
| `data/processed/reviews_sample.csv` | Anonymized sample of 500 reviews from notebook 01, used in notebooks 03 and 04. |
| `data/silver_reference.csv` | Prepared sample IDs and silver labels for 40 reviews, included in the Moodle data zip. |
| `data/processed/silver_labels.csv` | 40 reviews with prepared silver labels from `03_Classify_Reviews`, used in `05_Local_Model`. |

## Licence

The data comes from the [Yelp Open Dataset](https://yelp.com/dataset) and is licensed for non-commercial,
academic use only; it may not be published. That's why it is distributed on Moodle and
never stored in this repository. See [Data](#data) for the rules.
