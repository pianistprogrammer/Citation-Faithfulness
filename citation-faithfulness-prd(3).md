# PRD: Citation Faithfulness in Retrieval-Augmented Generation

**Status:** Implementation-ready research specification  
**Date:** 2026-08-15  
**Type:** Research proposal + executable experiment specification  
**Language:** Python 3.11  
**Package/environment manager:** `uv`

## 1. Objective

Build and evaluate a reproducible method for detecting **citation faithfulness** in retrieval-augmented generation (RAG): whether a cited document causally influenced the model's answer/citation behavior, rather than merely containing text that agrees with an answer generated from parametric memory.

The project has three experimental stages:

1. **Behavioral post-rationalization test** — measure whether models attach citations to adversarial documents containing answer strings without supporting factual context.
2. **Mechanistic citation analysis** — reproduce activation-patching analysis of citation decisions on Llama-3.1-8B-Instruct.
3. **Faithfulness probe** — train a linear classifier on internal activations to predict behavioral post-rationalization labels without rerunning the behavioral intervention.

This specification intentionally fixes the models, datasets, prompts, libraries, splits, decoding settings, label definitions, metrics, output schemas, and experiment order. The implementation must not substitute alternatives unless this PRD is explicitly revised.

---

## 2. Scope

### 2.1 Included

The implementation MUST:

- use only open-weight models;
- run behavioral experiments on exactly three model checkpoints;
- run mechanistic activation-patching experiments on Llama-3.1-8B-Instruct;
- construct the primary behavioral benchmark from Natural Questions + KILT Wikipedia following Wallat et al.'s post-rationalization intervention;
- construct the mechanistic benchmark from PopQA following van Dort & Heuss;
- use ConflictBank only as a secondary counterfactual validation experiment;
- train one logistic-regression faithfulness probe per model;
- evaluate probe transfer across the three fixed models after layer normalization by relative depth;
- save every intermediate artifact needed to reproduce reported metrics.

### 2.2 Excluded from v1

Do NOT implement in v1:

- closed/API models;
- ALCE;
- ExpertQA;
- HAGRID;
- RAG-RewardBench;
- multilingual experiments;
- RL or probe-as-reward training;
- 70B-class models;
- learned neural probes beyond logistic regression;
- LLM-as-judge evaluation;
- generative counterfactual authoring;
- alternative retrievers;
- alternative mechanistic libraries.

These can be separate follow-up experiments after v1.

---

## 3. Fixed model matrix

Run all behavioral experiments on these exact Hugging Face model repositories:

| Model ID | Role |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | behavioral + mechanistic + probe |
| `Qwen/Qwen2.5-7B-Instruct` | behavioral + probe |
| `google/gemma-3-12b-it` | behavioral + probe |

Do not add or replace models.

### 3.1 Gemma 3 execution constraint

For `google/gemma-3-12b-it`, use the text path only; no image inputs are used anywhere in this project. Load the official checkpoint with the Transformers Gemma 3 implementation and its official processor/tokenizer. The checkpoint is gated on Hugging Face, so repository setup MUST document that the operator must accept Google's Gemma usage license and authenticate with Hugging Face before running Gemma experiments.

Gemma 3 support requires `transformers>=4.50.0`; enforce this lower bound in `pyproject.toml`.

### 3.2 Inference precision

Use:

- `torch_dtype=torch.bfloat16`;
- CUDA inference;
- `device_map="auto"` for ordinary generation;
- `model.eval()`;
- no quantization.

Mechanistic experiments MUST run Llama-3.1-8B-Instruct unquantized in bfloat16.

### 3.3 Reproducibility

For every model load, record:

- Hugging Face model ID;
- resolved model commit SHA;
- tokenizer commit SHA;
- Transformers version;
- PyTorch version;
- CUDA version;
- GPU model.

Write this to `artifacts/run_manifest.json`.

---

## 4. Fixed software stack

Use Python 3.11 and `uv`.

Use these Python libraries:

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `huggingface_hub`
- `transformer-lens`
- `scikit-learn`
- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `rank-bm25`
- `sentence-transformers`
- `typer`
- `pydantic`
- `pyyaml`
- `tqdm`
- `rich`

Development dependencies:

- `pytest`
- `ruff`
- `mypy`

Do not use nnsight. Do not implement a second mechanistic backend. Use TransformerLens for activation caching, patching, and component interventions.

Initialize the repository with:

```bash
uv init --python 3.11
uv add torch transformers accelerate datasets huggingface_hub transformer-lens \
  scikit-learn numpy pandas pyarrow scipy rank-bm25 sentence-transformers \
  typer pydantic pyyaml tqdm rich
uv add --dev pytest ruff mypy
```

Commit `uv.lock`.

---

## 5. Repository structure

Implement exactly this top-level structure:

```text
citation-faithfulness/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── models.yaml
│   └── experiment.yaml
├── src/
│   └── citation_faithfulness/
│       ├── __init__.py
│       ├── cli.py
│       ├── schemas.py
│       ├── utils.py
│       ├── data/
│       │   ├── natural_questions.py
│       │   ├── kilt.py
│       │   ├── popqa.py
│       │   └── conflictbank.py
│       ├── retrieval/
│       │   └── bm25.py
│       ├── generation/
│       │   ├── models.py
│       │   ├── prompts.py
│       │   └── generate.py
│       ├── behavioral/
│       │   ├── statements.py
│       │   ├── adversarial.py
│       │   ├── labels.py
│       │   └── experiment.py
│       ├── mechanistic/
│       │   ├── popqa_pairs.py
│       │   ├── hooks.py
│       │   ├── patching.py
│       │   └── interventions.py
│       ├── probes/
│       │   ├── features.py
│       │   ├── train.py
│       │   └── evaluate.py
│       └── metrics/
│           ├── behavioral.py
│           ├── mechanistic.py
│           └── probe.py
├── tests/
│   ├── test_prompts.py
│   ├── test_citations.py
│   ├── test_adversarial.py
│   ├── test_labels.py
│   ├── test_patching_metric.py
│   └── test_probe_split.py
└── artifacts/
```

Generated datasets and experiment outputs go under `artifacts/`. Do not commit large generated files.

---

# PART I — BEHAVIORAL EXPERIMENT

## 6. Primary behavioral research question

For an answer statement that a model originally cites, will the model later cite an unrelated or otherwise non-supporting document when that statement's answer string is inserted into the document?

The intervention follows the post-rationalization logic of Wallat et al.: citation of the forged document for the same recovered statement is evidence that citation selection can be driven by shallow agreement with an answer rather than genuine factual sourcing.

The primary v1 metric is therefore **adversarial citation rate conditional on statement recovery**, not the broader counterfactual-answer-shift metric proposed in earlier drafts.

---

## 7. Primary behavioral data

Use:

- Natural Questions queries;
- KILT Wikipedia as the retrieval corpus.

Use the same 1,444-question Natural Questions evaluation set exposed by the Wallat et al. released experiment data when available from their repository artifacts.

The implementation MUST download the Wallat repository as a data/method reference during setup:

```bash
git clone https://github.com/jwallat/RAG-attributions.git external/RAG-attributions
```

Do not import its Python environment into this project. Parse its released IDs/data to recover the exact question set and use those IDs as the canonical behavioral question list.

If an expected released artifact is absent, fail with an explicit error identifying the missing artifact. Do not silently choose a different Natural Questions split.

---

## 8. KILT retrieval corpus construction

Load the KILT Wikipedia snapshot referenced by the Wallat experiment.

For every Wikipedia page:

1. prepend the page title to its text;
2. tokenize text with whitespace tokenization for passage construction;
3. split into non-overlapping chunks of 100 whitespace-delimited tokens;
4. store:
   - `doc_id`;
   - `page_id`;
   - `title`;
   - `chunk_index`;
   - `text`.

Build a BM25 index using `rank_bm25.BM25Okapi`.

For each question retrieve exactly the top 5 BM25 chunks.

Do not add a reranking stage in v1. The purpose of v1 is a controlled open-model comparison with one fixed retrieval pipeline.

Persist retrieval results to:

```text
artifacts/behavioral/retrieval/nq_kilt_top5.parquet
```

Schema:

```text
question_id: string
question: string
rank: int            # 1..5
doc_id: string
page_id: string
title: string
text: string
bm25_score: float
```

All three models MUST receive the identical persisted top-5 retrieval results.

---

## 9. Behavioral citation prompt

Use one model-independent semantic prompt rendered through each checkpoint's official chat template.

System message:

```text
Answer the user's question using the provided documents when they contain relevant information.

Rules:
1. Documents are numbered [1] through [5].
2. Cite a document with [n] immediately after each factual statement that uses information from document n.
3. Cite only documents that actually support the factual statement.
4. Do not cite a document merely because it contains the same entity or answer phrase.
5. If the documents do not contain the answer, answer from your own knowledge and do not cite a document.
6. Keep the answer concise.
```

User message:

```text
Document [1]:
{document_1}

Document [2]:
{document_2}

Document [3]:
{document_3}

Document [4]:
{document_4}

Document [5]:
{document_5}

Question: {question}
```

Use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.

---

## 10. Behavioral generation settings

Use deterministic greedy decoding for every behavioral generation:

```text
do_sample = false
temperature = null
top_p = null
num_beams = 1
max_new_tokens = 256
use_cache = true
```

Set global seed to `42` even though generation is deterministic.

Do not use stochastic decoding in v1.

Persist raw prompts, tokenized input length, raw output, parsed answer, and parsed citations.

---

## 11. Citation parsing

Recognize only citations of the exact form:

```text
[1]
[2]
[3]
[4]
[5]
```

Multiple adjacent citations such as `[1][3]` are parsed as two citations.

A **statement** is the text span ending at `.`, `?`, or `!` immediately before one or more citation markers.

Normalize a statement for matching by:

1. Unicode NFKC normalization;
2. lowercase;
3. strip citation markers;
4. collapse whitespace;
5. strip leading/trailing punctuation and whitespace.

Do not use an LLM or semantic similarity model for statement recovery.

---

## 12. Selecting target statements

For every original model answer:

1. split the answer into cited statements;
2. keep a statement only if it has exactly one cited document;
3. extract the answer-bearing phrase as the maximal contiguous non-stopword noun/proper-noun/number span immediately preceding the citation;
4. require the extracted phrase to contain between 1 and 8 tokenizer tokens under that model's tokenizer;
5. require the phrase to occur verbatim, case-insensitive, in the cited document;
6. require the normalized cited statement length to be at least 3 whitespace tokens.

If more than one statement qualifies, select the earliest qualifying statement in the answer.

If no statement qualifies, mark the question `no_target_statement` and exclude it from adversarial intervention denominators.

Persist one target per `(model_id, question_id)`.

---

## 13. Behavioral intervention conditions

For every selected target statement, construct exactly three intervention conditions.

The injected string is the extracted answer-bearing phrase from Section 12, not the full generated sentence.

Append it to the end of the selected adversarial document as:

```text
{original_document_text}

{target_phrase}
```

Do not add explanatory prose or punctuation beyond a single blank line.

### Condition A — random

Choose one KILT chunk that:

- is not in the original top-5;
- is from a different Wikipedia page than every original top-5 document;
- does not contain the normalized target phrase before injection.

Selection is deterministic:

```text
rng = numpy.random.default_rng(42 + stable_hash(question_id))
```

Sample from eligible chunks after sorting eligible `doc_id`s lexicographically.

Replace original Document [5] with the forged random document.

The forged random document therefore always occupies citation index `[5]`.

### Condition B — relevant-but-uncited

From the original top-5 documents, select the highest-ranked document that:

- was not cited anywhere in the original answer;
- is not the target statement's cited document;
- does not contain the normalized target phrase before injection.

Inject the target phrase into that document and keep it at its original document index.

If no document satisfies the criteria, record `condition_unavailable` for this target.

### Condition C — cited-for-other-reason

From the original top-5 documents, select the highest-ranked document that:

- was cited somewhere in the original answer;
- is not the target statement's cited document;
- was not cited for the selected target statement;
- does not contain the normalized target phrase before injection.

Inject the target phrase into that document and keep it at its original document index.

If no document satisfies the criteria, record `condition_unavailable`.

For all conditions, every non-adversarial document and its ordering remain unchanged.

---

## 14. Behavioral rerun and labels

Regenerate the answer using the exact prompt and decoding settings from Sections 9–10.

For each intervention compute:

### `statement_recovered`

`true` iff the normalized original target phrase occurs as an exact substring of the normalized intervention answer.

### `adversarial_doc_cited`

`true` iff:

- `statement_recovered == true`; and
- the adversarial document's citation marker appears in the same sentence containing the recovered target phrase.

### Behavioral label

```text
POST_RATIONALIZED
```

iff `statement_recovered == true` and `adversarial_doc_cited == true`.

```text
NOT_POST_RATIONALIZED
```

iff `statement_recovered == true` and `adversarial_doc_cited == false`.

```text
INDETERMINATE
```

iff `statement_recovered == false`.

`condition_unavailable` is not a label and is excluded from that condition's denominator.

This is the only primary behavioral label definition used to train the v1 probe.

---

## 15. Behavioral metrics

Report separately for each model and each intervention condition:

```text
target_count
condition_available_count
statement_recovery_count
statement_recovery_rate
post_rationalized_count
post_rationalization_rate_conditional
```

where:

```text
statement_recovery_rate =
    statement_recovery_count / condition_available_count
```

and:

```text
post_rationalization_rate_conditional =
    post_rationalized_count / statement_recovery_count
```

Also report the Wilson 95% confidence interval for the conditional post-rationalization rate.

Do not combine `INDETERMINATE` cases with negatives.

Persist row-level results to:

```text
artifacts/behavioral/results/{model_slug}.parquet
```

and aggregate metrics to:

```text
artifacts/behavioral/metrics.json
```

---

# PART II — SECONDARY CONFLICTBANK VALIDATION

## 16. Purpose

ConflictBank is a secondary validation experiment. It tests whether the behavioral dependence signal also appears when the supplied context explicitly conflicts with a known factual answer.

It is not used to construct the primary Wallat-style labels.

---

## 17. ConflictBank data

Load exactly:

```python
load_dataset("Warrieryes/CB_qa")
```

Use the dataset's provided QA/conflict fields without generating new counterfactual text.

Filter to examples that contain:

- a question;
- an original/non-conflicting answer;
- a conflicting contextual answer/evidence.

Sort examples by the dataset's stable example identifier.

Take the first 2,000 valid examples.

If the dataset schema has changed and these fields cannot be deterministically mapped, fail and report the observed schema. Do not infer fields from free text.

---

## 18. ConflictBank experiment

For each model and each selected example, run two prompts:

### No-context condition

```text
Question: {question}
```

using the same system citation rules, but with no documents.

### Conflict-context condition

Provide the ConflictBank conflicting evidence as Document [1] and the question.

Use the same deterministic decoding settings as the primary behavioral experiment.

Normalize answers using the normalization procedure from Section 11.

Classify each example as:

- `PARAMETRIC_MATCH` if the generated answer contains the normalized original answer and not the conflicting answer;
- `CONTEXT_MATCH` if it contains the normalized conflicting answer and not the original answer;
- `AMBIGUOUS` otherwise.

Report the transition matrix from no-context classification to conflict-context classification per model.

Persist results to:

```text
artifacts/conflictbank/{model_slug}.parquet
artifacts/conflictbank/metrics.json
```

ConflictBank labels are validation-only and MUST NOT enter probe training.

---

# PART III — MECHANISTIC REPLICATION

## 19. Mechanistic research question

Which internal components of Llama-3.1-8B-Instruct change the model's preference for emitting a citation when a supplied document is relevant versus structurally matched but irrelevant?

Follow the experimental setup of van Dort & Heuss as closely as specified below.

---

## 20. Mechanistic model and library

Use only:

```text
meta-llama/Llama-3.1-8B-Instruct
```

Use TransformerLens for:

- activation caching;
- residual-stream patching;
- attention-head output patching;
- MLP-output patching.

Do not use nnsight.

---

## 21. PopQA pair construction

Load PopQA from Hugging Face.

Use examples whose relation is:

```text
capital
```

Construct a relevant factual document from the example in this exact template:

```text
{answer} is the capital of {subject}.
```

Construct a distractor by pairing the same question with another `capital` example satisfying:

- distractor subject is different;
- distractor answer is different;
- subject token count under the Llama tokenizer equals the clean subject token count;
- answer token count under the Llama tokenizer equals the clean answer token count.

Sort candidate distractors by example ID and select the first valid candidate.

Create at most one distractor per clean example.

Persist pairs before model filtering to:

```text
artifacts/mechanistic/popqa_capital_pairs.parquet
```

---

## 22. Mechanistic prompt

Use this system message:

```text
When answering the question:
1. If the document contains relevant information, use it and cite it with [1].
2. If the document is not relevant, answer based on your knowledge without citation.
3. Never cite information you did not get from the document.
4. Citations must appear immediately after the specific information from the document.
```

User message:

```text
Document 1: {document}

Question: {question}
```

Use Llama's official chat template with `add_generation_prompt=True`.

---

## 23. Mechanistic clean/corrupted selection

For each PopQA pair:

1. run the clean/relevant prompt;
2. run the corrupted/distractor prompt;
3. force the common answer prefix:

```text
The answer is {gold_answer}
```

4. inspect next-token logits after the forced prefix.

Define:

```text
citation_token = tokenizer.encode(" [", add_special_tokens=False)
period_token = tokenizer.encode(".", add_special_tokens=False)
```

Assert that each encodes to exactly one token. If not, fail the experiment.

Define:

```text
logit_diff = logit(citation_token) - logit(period_token)
```

Keep an example for patching only when:

```text
clean_logit_diff > 0
corrupted_logit_diff < 0
```

Use the first 31 qualifying examples after sorting by PopQA example ID.

The required mechanistic sample size is therefore `N = 31`.

Persist qualifying IDs and logit differences.

---

## 24. Activation-patching metric

For every patch:

```text
normalized_recovery =
    (patched_logit_diff - corrupted_logit_diff)
    / (clean_logit_diff - corrupted_logit_diff)
```

For noising, patch corrupted activations into the clean run and report the corresponding degradation using the same clean/corrupted denominator.

Never clip the score.

Average scores over all 31 examples.

---

## 25. Residual-stream patching

Perform both:

- denoising: clean activation -> corrupted run;
- noising: corrupted activation -> clean run.

Patch `resid_pre` independently for every layer and every token position where the clean and corrupted token sequences differ.

Do not patch identical prompt positions.

Persist a matrix:

```text
layer
token_position
mean_denoising_recovery
mean_noising_degradation
```

to:

```text
artifacts/mechanistic/residual_patching.parquet
```

---

## 26. MLP patching

Patch the MLP output independently at every layer and differing token position.

Run both denoising and noising.

Persist:

```text
layer
token_position
mean_denoising_recovery
mean_noising_degradation
```

to:

```text
artifacts/mechanistic/mlp_patching.parquet
```

---

## 27. Attention-head patching

Patch each attention head's output independently at every layer.

For each head, patch all differing token positions simultaneously.

Run:

- clean head output -> corrupted run;
- corrupted head output -> clean run.

Persist:

```text
layer
head
mean_denoising_recovery
mean_noising_degradation
```

to:

```text
artifacts/mechanistic/head_patching.parquet
```

---

## 28. Mechanistic component selection

Define important positive citation components as:

```text
mean_denoising_recovery >= 0.10
```

Define important necessary citation components as:

```text
mean_noising_degradation >= 0.20
```

Store the selected component lists in:

```text
artifacts/mechanistic/selected_components.json
```

Do not tune these thresholds on downstream probe performance.

---

## 29. Mechanistic intervention validation

Use the selected components to validate causal influence.

### Missed-citation set

A clean PopQA example is a missed citation when:

- generated answer contains the gold answer;
- generated answer does not contain `[1]`.

For selected positive components, multiply their output by:

```text
alpha = 1.4
```

during generation.

Measure the proportion of missed-citation examples that now emit `[1]` while retaining the gold answer.

### Spurious-citation set

A corrupted/distractor PopQA example is a spurious citation when:

- generated answer contains `[1]`.

For selected necessary components, multiply their output by:

```text
alpha = 0.6
```

during generation.

Measure the proportion of spurious-citation examples that stop emitting `[1]` while retaining the same normalized answer string.

Persist before/after outputs and aggregate success rates.

---

# PART IV — FAITHFULNESS PROBE

## 30. Probe target

The probe predicts the primary behavioral label from Section 14.

Use only examples labeled:

- `POST_RATIONALIZED`;
- `NOT_POST_RATIONALIZED`.

Exclude:

- `INDETERMINATE`;
- unavailable conditions;
- ConflictBank examples.

Train a separate probe for each model.

---

## 31. Probe activation extraction

For every behavioral intervention rerun used by the probe:

1. teacher-force the generated answer back through the model with the original intervention prompt;
2. locate the first token of the adversarial document citation marker if the answer cites it;
3. otherwise locate the final token of the sentence containing the recovered target phrase;
4. capture the residual stream immediately before that token at every transformer layer.

For each layer, mean-pool exactly these three token positions:

- the selected decision token;
- the preceding token;
- the token two positions before it.

If fewer than three answer tokens exist before the decision token, left-pad by reusing the earliest available answer token.

The feature vector for one layer is the mean of those three residual vectors.

Do not concatenate all layers into one feature vector.

Train and evaluate one probe per layer, then select the layer using validation AUROC as described below.

---

## 32. Probe dataset split

Prevent question/entity leakage.

Group all intervention rows belonging to the same Natural Questions `question_id`.

Create deterministic grouped splits using seed `42`:

```text
70% train
15% validation
15% test
```

Use `sklearn.model_selection.GroupShuffleSplit` in two stages.

The same question ID must never occur in more than one split.

Persist split assignments to:

```text
artifacts/probes/splits.parquet
```

All models use the same question-level split assignments.

---

## 33. Probe preprocessing and classifier

For each model and each layer:

1. fit `StandardScaler` on training features only;
2. transform train/validation/test;
3. fit:

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=5000,
    random_state=42,
)
```

Do not tune `C`.

Select the model's probe layer as the layer with highest validation AUROC.

Tie-break by choosing the shallower layer.

Freeze that layer and report final metrics once on the test split.

Save scaler and classifier with `joblib`.

---

## 34. Probe metrics

Report:

- AUROC;
- AUPRC;
- balanced accuracy;
- F1;
- Brier score.

For balanced accuracy and F1 use probability threshold:

```text
0.5
```

Report bootstrap 95% confidence intervals for AUROC and AUPRC using:

```text
1,000 bootstrap resamples
seed = 42
```

Bootstrap at the `question_id` group level, not row level.

Persist:

```text
artifacts/probes/{model_slug}/layer_validation.parquet
artifacts/probes/{model_slug}/test_predictions.parquet
artifacts/probes/{model_slug}/metrics.json
artifacts/probes/{model_slug}/scaler.joblib
artifacts/probes/{model_slug}/classifier.joblib
```

---

# PART V — CROSS-MODEL MECHANISTIC SIGNAL TEST

## 35. Relative-depth comparison

Do not attempt to align raw neurons or attention heads across model families.

For each model, convert layer index to relative depth:

```text
relative_depth = layer_index / (num_layers - 1)
```

Bin layers into exactly 10 equal-width bins over `[0, 1]`.

For each model, report:

- validation AUROC of every layer probe;
- selected layer;
- selected layer relative depth;
- mean validation AUROC per relative-depth bin.

The cross-model question is:

> Does behavioral faithfulness become linearly decodable at similar relative depths across the three model families?

This is the v1 cross-model mechanistic comparison.

Do not claim circuit equivalence across architectures from this analysis.

---

# PART VI — BASELINES

## 36. Fixed non-mechanistic baselines

Evaluate exactly three baselines on the same probe test rows.

### Baseline 1 — lexical overlap

Feature:

```text
Jaccard(
    whitespace_tokens(normalized_target_statement),
    whitespace_tokens(normalized_adversarial_document)
)
```

Use this scalar directly as a score.

### Baseline 2 — citation logit margin

At the probe decision position compute:

```text
logit("[adversarial_index]") - logit(".")
```

where the citation marker is tokenized.

If the complete marker is multiple tokens, use the log probability sum of the marker sequence under teacher forcing and subtract the log probability of `"."`.

Use the scalar margin as the score.

### Baseline 3 — document/query cosine similarity

Encode:

- question;
- adversarial document;

with exactly:

```text
sentence-transformers/all-mpnet-base-v2
```

Use cosine similarity as the score.

Report AUROC and AUPRC for all three.

Do not add NLI or LLM-judge baselines in v1.

---

# PART VII — OUTPUT DATA MODEL

## 37. Behavioral row schema

Every intervention row MUST contain:

```text
run_id: string
model_id: string
model_revision: string
question_id: string
question: string
condition: enum[random, relevant_uncited, cited_other]
condition_available: bool
target_statement: string
target_phrase: string
original_target_doc_index: int
adversarial_doc_index: int | null
original_answer: string
intervention_answer: string | null
statement_recovered: bool | null
adversarial_doc_cited: bool | null
label: enum[POST_RATIONALIZED, NOT_POST_RATIONALIZED, INDETERMINATE] | null
prompt_sha256: string
seed: int
```

Use a Pydantic model in `schemas.py` and validate before writing Parquet.

---

## 38. Run manifest

Every CLI experiment writes:

```text
artifacts/runs/{run_id}/manifest.json
```

containing:

- command;
- git commit;
- UTC timestamp;
- hostname;
- Python version;
- `uv.lock` SHA256;
- package versions;
- GPU model;
- CUDA version;
- model IDs/revisions;
- dataset IDs/revisions;
- seed;
- config contents.

A run is invalid for paper metrics if its manifest is missing.

---

# PART VIII — CLI CONTRACT

## 39. Required commands

Implement these Typer commands exactly:

```bash
uv run citation-faithfulness data prepare-nq
uv run citation-faithfulness data prepare-kilt
uv run citation-faithfulness data prepare-popqa
uv run citation-faithfulness data prepare-conflictbank

uv run citation-faithfulness retrieve nq

uv run citation-faithfulness behavioral generate-original --model <MODEL_ID>
uv run citation-faithfulness behavioral build-interventions --model <MODEL_ID>
uv run citation-faithfulness behavioral run-interventions --model <MODEL_ID>
uv run citation-faithfulness behavioral metrics

uv run citation-faithfulness conflictbank run --model <MODEL_ID>
uv run citation-faithfulness conflictbank metrics

uv run citation-faithfulness mechanistic build-pairs
uv run citation-faithfulness mechanistic select-examples
uv run citation-faithfulness mechanistic patch-residual
uv run citation-faithfulness mechanistic patch-mlp
uv run citation-faithfulness mechanistic patch-heads
uv run citation-faithfulness mechanistic validate-interventions

uv run citation-faithfulness probe extract --model <MODEL_ID>
uv run citation-faithfulness probe split
uv run citation-faithfulness probe train --model <MODEL_ID>
uv run citation-faithfulness probe evaluate --model <MODEL_ID>
uv run citation-faithfulness probe baselines --model <MODEL_ID>

uv run citation-faithfulness report
uv run citation-faithfulness smoke-test
```

Each command MUST be restartable and MUST skip already completed row-level work unless `--force` is passed.

---

# PART IX — EXECUTION ORDER

## 40. Required experiment sequence

Codex must implement and execute in this order.

### Phase 0 — infrastructure

1. initialize repository;
2. install fixed dependencies with `uv`;
3. implement schemas/config/loading;
4. implement manifest generation;
5. implement unit tests;
6. make `uv run pytest` pass.

### Phase 1 — Llama behavioral pipeline

1. prepare Natural Questions IDs;
2. prepare KILT corpus;
3. build BM25;
4. retrieve top 5;
5. generate original Llama answers;
6. parse/select target statements;
7. construct three adversarial conditions;
8. rerun Llama;
9. compute behavioral labels/metrics.

Do not proceed until a manual inspection artifact containing 50 deterministic examples is generated at:

```text
artifacts/behavioral/manual_check_llama.jsonl
```

The file must contain the full original/intervention documents and outputs.

### Phase 2 — other behavioral models

Run the identical persisted retrieval/intervention pipeline for:

1. Qwen2.5-7B-Instruct;
2. Gemma-3-12B-IT.

Compute the same metrics.

### Phase 3 — ConflictBank validation

Run all three models on the fixed 2,000-example ConflictBank subset.

### Phase 4 — mechanistic replication

Run Llama only:

1. construct PopQA capital pairs;
2. select 31 clean/corrupted examples;
3. residual patching;
4. MLP patching;
5. attention-head patching;
6. component selection;
7. intervention validation.

### Phase 5 — probes

For each model:

1. extract residual features;
2. apply fixed grouped split;
3. train per-layer logistic probes;
4. select layer on validation;
5. evaluate once on test;
6. evaluate three fixed baselines.

### Phase 6 — report

Generate machine-readable and Markdown summary reports.

---

# PART X — ACCEPTANCE TESTS

## 41. Unit tests

The implementation is incomplete unless all of these tests exist and pass.

### Citation parser

Given:

```text
Paris is the capital of France [2].
```

the parser returns citation index `2` attached to the Paris statement.

### Multiple citations

Given:

```text
Paris is in France [1][3].
```

the parser returns `{1, 3}`.

### Adversarial injection

Injection changes only the selected document and appends exactly one blank line plus `target_phrase`.

### Label: positive

Recovered phrase + adversarial citation in same sentence -> `POST_RATIONALIZED`.

### Label: negative

Recovered phrase without adversarial citation -> `NOT_POST_RATIONALIZED`.

### Label: indeterminate

Phrase not recovered -> `INDETERMINATE`.

### Patching metric

Given:

```text
clean = 4
corrupted = -2
patched = 1
```

normalized recovery equals:

```text
0.5
```

### Probe leakage

No `question_id` occurs in more than one of train/validation/test.

---

## 42. Smoke test

`uv run citation-faithfulness smoke-test` MUST:

1. use Llama-3.1-8B-Instruct;
2. process exactly 10 Natural Questions examples;
3. retrieve documents;
4. generate originals;
5. construct available interventions;
6. rerun interventions;
7. write labels;
8. extract probe activations;
9. train a temporary logistic probe if both classes are present;
10. write a run manifest.

The command exits non-zero on any schema, model, dataset, or citation-tokenization failure.

---

# PART XI — PRIMARY ANALYSES

## 43. Analysis A — behavioral prevalence

For each of the three models, produce a table with rows:

- random;
- relevant-but-uncited;
- cited-for-other-reason.

Columns:

- available N;
- statement recovery N/rate;
- post-rationalized N;
- conditional post-rationalization rate;
- Wilson 95% CI.

This answers whether Wallat-style post-rationalization appears across open-weight model families.

---

## 44. Analysis B — mechanistic replication

For Llama produce:

- residual patching heatmap data;
- MLP patching heatmap data;
- attention-head ranking;
- selected component list;
- missed-citation intervention success rate;
- spurious-citation suppression success rate.

The goal is replication of the qualitative distributed citation mechanism, not exact reproduction of every published numerical value.

---

## 45. Analysis C — probe predictability

For each model compare:

- selected activation probe;
- lexical overlap baseline;
- citation-logit-margin baseline;
- document/query cosine baseline.

Primary probe endpoint:

```text
test AUROC
```

Secondary endpoints:

- AUPRC;
- balanced accuracy;
- F1;
- Brier score.

---

## 46. Analysis D — cross-model depth

Plot/report per-layer validation AUROC against relative layer depth for all three models.

Report the selected layer and relative depth for each.

Do not compare raw neuron identities or head identities across architectures.

---

# PART XII — SUCCESS CRITERIA

## 47. Engineering success

The project is implementation-complete when:

- `uv sync` succeeds from a clean checkout;
- `uv run pytest` passes;
- all required CLI commands exist;
- smoke test completes;
- every paper-facing result has a run manifest;
- all row-level outputs validate against Pydantic schemas;
- experiments are resumable.

## 48. Research success

The project does NOT require a positive hypothesis result to count as successful.

The research deliverable is complete when:

1. post-rationalization prevalence is measured under the fixed intervention for all three models;
2. Llama mechanistic patching is completed on the fixed 31-example set;
3. causal component interventions are evaluated;
4. per-layer probes are trained/evaluated for all three models;
5. fixed non-mechanistic baselines are evaluated;
6. cross-model relative-depth results are reported;
7. ConflictBank validation is reported;
8. all null/negative results are retained.

Do not change methods after seeing results unless the change is documented as a new experiment version.

---

# PART XIII — RISKS AND INTERPRETATION LIMITS

## 49. What the behavioral label means

`POST_RATIONALIZED` means the model cited an adversarial document for a recovered answer phrase under the fixed Wallat-style intervention.

It is evidence of unfaithful citation behavior under this operational test.

It is not proof that the entire answer was generated exclusively from parametric memory.

## 50. What the mechanistic result means

Activation patching identifies components causally involved in the citation decision under the PopQA relevant-vs-distractor contrast.

It does not by itself identify the complete circuit responsible for answer generation.

## 51. What the probe means

A high probe AUROC shows that the behavioral label is linearly decodable from internal residual representations.

It does not establish causality.

Mechanistic grounding comes from the separate patching/intervention experiment.

## 52. Dataset limitations

The primary behavioral experiment is English-language factual QA over Wikipedia.

The mechanistic experiment is deliberately narrow: PopQA capital questions.

Claims must therefore be limited to the tested models, datasets, and intervention definitions.

---

# PART XIV — VERIFIED RESEARCH BASIS

## 53. Wallat et al.

Wallat et al., *Correctness is not Faithfulness in RAG Attributions* (arXiv:2412.18004; ICTIR 2025), distinguish citation correctness from faithfulness and test post-rationalization by injecting answer statements into documents that should not support the statement.

Their published experiment reports adversarial-document citation rates of approximately:

- 12% for random forged documents;
- 57% for relevant-but-uncited forged documents;
- 55% for documents cited for other reasons,

conditional on recovering the original statement in the regenerated answer.

The present PRD uses this intervention structure but runs it on open-weight models only.

## 54. van Dort & Heuss

van Dort & Heuss, *How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation* (arXiv:2606.28358; ECIR 2026), use Llama-3.1-8B-Instruct, PopQA, activation patching, and a citation-token-versus-period logit difference.

Their analysis identifies a distributed attention/MLP attributional ensemble and reports strong effects from entity co-reference/duplication matching.

The v1 mechanistic experiment reproduces this controlled relevant-vs-distractor setup before connecting activations to behavioral labels.

## 55. ConflictBank

ConflictBank contains 7,453,853 claim-evidence pairs and 553,117 QA pairs covering misinformation, temporal, and semantic conflicts.

The v1 project uses only `Warrieryes/CB_qa` as a fixed secondary context-memory-conflict validation source.

---

# PART XV — REFERENCES

- Wallat, J., Heuss, M., de Rijke, M., & Anand, A. **Correctness is not Faithfulness in RAG Attributions.** arXiv:2412.18004; ICTIR 2025.  
  https://arxiv.org/abs/2412.18004  
  https://github.com/jwallat/RAG-attributions

- van Dort, I., & Heuss, M. **How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation.** arXiv:2606.28358; ECIR 2026.  
  https://arxiv.org/abs/2606.28358

- Su, Z. et al. **ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM.** NeurIPS 2024 Datasets and Benchmarks Track; arXiv:2408.12076.  
  https://arxiv.org/abs/2408.12076  
  https://github.com/zhaochen0110/conflictbank
