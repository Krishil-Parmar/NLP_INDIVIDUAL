# Router Before the LLM

A cost-aware intent router for a personal finance assistant. Individual university NLP project by Krishil Parmar.

This README is the single source of truth for the project. It is written so that Claude Code (or any new collaborator) can pick the work up cold. Read all of it before writing code.

Last updated: 23 Nov 2026. Status: **M4b (E2, label noise) done on validation, audit included. Every core-path item in section 0 works on validation (M4).** S0 to S3 run on Banking77 validation (five seeds for S0, S1 and S3; S2 has no training, so one run). S4 (Claude Sonnet 4.6 on Bedrock, zero-shot and few-shot) is cached in `results/s4_cache/`. E1 is in `results/e1_val.md`, E3 in `results/e3_val_fmt.md`, and E6 (policy v1, risk-coverage and cost, the headline) in `results/e6_val_fmt.md`, with thresholds and temperatures frozen in `results/policy_v1.json`. E2 is in `results/e2_val.md`: cleanlab flagged 192 train and validation rows, every system was retrained without them, and the system ranking did not change; the blind audit (owner items 1 to 11, a Claude judge items 12 to 100) finds 61 of 100 flagged items are real label errors (95% Wilson interval 51 to 70%). The notebook runs end to end with all of it and never calls the API. The two-page summary skeleton builds to 2 pages. The demo router server (`demo/app.py`) answers typed queries offline; its Jeff screen front end builds and passes `flutter analyze` but has not yet been clicked through in a browser. The Jeff set is built (section 6.1: 320 queries written by a Claude agent at the owner's request) and the thresholds are drafted in `results/preregistration_draft.md`. Next: the owner confirms the draft and the M5 S4 spend, then M5 (the single test run).

---

## 0. Marking rubric alignment (read this first)

Source: "Individual Project Evaluation Criteria (updated)", supplied by the owner on 22 Nov 2026. 100 points, five categories of 20 points each.

**What is submitted:** a 1 to 2 page summary report and the code. **What is shown:** a brief demo of the working application. No presentation is required.

| Category (20 points each) | What the rubric measures | Where this project earns it | What goes beyond the minimum |
| --- | --- | --- | --- |
| 1. Problem Understanding and Requirement Coverage | Clear explanation of the problem. Understanding of constraints and user needs. Evidence that the project actually solves the intended task. | Sections 1, 2.3 and 2.5 (requirements table, each requirement tied to an experiment) | A real product use case with a hard budget and latency constraint. Hypotheses written down before the test runs (section 4). |
| 2. Model and Dataset Selection | Appropriateness of chosen model. Appropriateness of dataset. Preprocessing pipeline quality. | Sections 6, 6.2, 7 and 7.2 | Five systems compared through one interface. Label-noise audit. A domain-shift set built from the assistant's real tool belt. |
| 3. Implementation Quality | Code correctness. Clear modular structure. Notebook runs end to end without breaking. | Sections 9, 10 and 10.1 | Unit tests for metrics and policy. Pinned dependencies. Clean-run check on a fresh machine before submission. |
| 4. Evaluation | Correct use of appropriate metrics. For classification: accuracy, F1, precision, recall. | Section 8 (E1) and section 8.1 | Five seeds and confidence intervals. Out-of-scope metrics. Risk-coverage and cost. Error analysis by reading examples. |
| 5. Final Demo and Communication | Clear, engaging demonstration. Clarity on what it does, why it works and its limitations. | Sections 12.1 and 12.2 | The demo shows the matched example behind each decision, a running count of LLM calls saved, and one failure case on purpose. |

Three consequences that override anything else in this file:

1. **The marker reads two pages, runs one notebook and watches one short demo.** Work that is not visible in one of those three places earns nothing. Every experiment must end as one table row, one figure or one sentence in the summary.
2. **40 of the 100 points (Implementation, Demo) depend on things working on the day.** A thin version that runs end to end is built first and kept working from then on. Depth is added on top of it, never instead of it.
3. **The rubric names accuracy, F1, precision and recall.** Those four go first in the results table. Coverage at matched risk is the extension and is explained in the same vocabulary (section 8.1).

**Core path. Finish this and keep it working before any stretch work:**

- Banking77 through the preprocessing pipeline (section 6.2)
- S0, S1 and S3 (with S2 as its retriever)
- E1 with accuracy, precision, recall and F1
- Router policy v1 and E6, using cached S4 responses
- The submission notebook (section 10.1)
- The demo (section 12.2)
- The two-page summary (section 12.1)

**Stretch. This is the "exceeds expectations" layer:** E2, E3, E4, E5, E7, policy v2 and CLINC150 as a second benchmark. Each gets at most one sentence or one table row in the summary. Details go in `report/appendix.md`.

---

## 1. North Star

**Remove as many LLM calls from a finance assistant as possible without making it worse.**

A small local model looks at each user query and makes one of three decisions:

| Decision | Meaning | Cost |
| --- | --- | --- |
| HANDLE | Router is confident about the intent and dispatches it directly | Near zero |
| REJECT | Query is clearly out of scope, return a canned response | Near zero |
| ESCALATE | Router is unsure, pass the query to the LLM | Full LLM cost |

**The one research question:** how many LLM calls can a small router safely remove, and at what error cost?

**The headline metric: coverage at matched risk.** The largest share of queries the router can decide by itself while its error rate on those queries stays at or below the error rate of the LLM baseline on the same queries.

**The headline chart:** a risk-coverage curve per system, with a second axis showing LLM calls, dollars and milliseconds saved at each threshold.

Every piece of work must move the headline metric, explain it, make it more trustworthy, or be a rubric deliverable from section 0 (notebook, demo, summary). If a task does none of these, do not do it.

---

## 2. Why this project exists

### 2.1 Coursework framing

The unit offers 11 individual project topics. This project is anchored on:

- **Topic 7, Intent Classification for Chatbots** ("build an intent detector for a domain, e.g. banking"). This is the official topic.
- **Topic 2, Paraphrase Detection** ("binary classifier for whether two sentences mean the same thing"). This is used as the method, not as a second project.

### 2.2 Why it is not the generic version

The generic version is: fine-tune BERT on Banking77, report about 93% accuracy, done. That result is weak because:

1. **The benchmark is noisy.** Ying and Thomas (2022) flagged over 1,400 of the 10,003 Banking77 training utterances (about 14%) as possibly mislabelled.
2. **In-scope accuracy is the easy part.** Larson et al. (2019) showed classifiers do well in scope and struggle to reject out-of-scope queries.
3. **A fixed softmax cannot take new intents without retraining.** Pairwise matching can. Zhang et al. (2020) got 10-shot performance competitive with 50-shot and full-data classifiers this way.
4. **Nobody else will report cost.** In 2026 the obvious objection to any classifier is "why not just prompt an LLM?". This project answers it with numbers instead of ignoring it.

### 2.3 The real use case

The owner runs a personal finance app whose assistant ("Jeff") is LLM-backed, has a hard per-user inference budget, a sub-900ms time-to-first-token target and a small tool belt. Every query that reaches the LLM spends budget. A router that safely absorbs the easy queries has direct product value. The project is designed so the result can be used there, but it must stand alone as coursework using public data.

### 2.4 What was considered and rejected

- **Hate speech plus emotion ("PRISM").** Rejected: the most over-subscribed pairing on the topic list, six loosely connected components, and its central finding (fewer false positives from emotion multi-task learning) is already published.
- **Adding emotion detection (topic 5) to this router.** Rejected: scope sprawl. Do not add it back.

### 2.5 Requirements and how each is checked

This table is the evidence for rubric category 1. Each user or product need becomes a requirement, and each requirement is checked by a named experiment. The summary report states, for each one, met or not met, with the number.

| ID | User or product need | Requirement | Checked by |
| --- | --- | --- | --- |
| R1 | A user asks a banking question and is routed to the right action | Correct intent on in-scope queries | E1 |
| R2 | A user is never given a confident answer to something the assistant cannot do | Out-of-scope and uncertain queries are not handled directly | E3, and the confident wrong HANDLE rate in E6 |
| R3 | The per-user inference budget is hard | Fewer LLM calls at an error rate no higher than LLM-only | E6 |
| R4 | Time to first token must stay under 900ms | The router adds little latency. Report p50 and p95 on CPU against a limit of [L] ms set by the owner. | E6 latency |
| R5 | The assistant's tool belt changes over time | A new intent can be added from about 10 examples with no retraining | E4 |
| R6 | Users phrase the same request in many ways | Decisions are stable under rewording | E5 |
| R7 | Real users are Australian and do not write like Banking77 | The router still works on queries written for the assistant's own intents (the Jeff set, section 6.1) | E7 |

R1 to R4 are on the core path. R5 to R7 are stretch.

---

## 3. Honest positioning

None of the components here is new research. Pairwise few-shot intent detection, out-of-scope detection, confident learning and selective classification all exist. The contribution is the applied question, the evaluation design, the cost analysis and a small domain-shift test set built from the assistant's real tool belt. Do not write "novel architecture" anywhere in the report.

---

## 4. Research questions and hypotheses

The owner must fill in the bracketed thresholds **before** any test-set run. Record them in `results/preregistration.md` with a date.

| ID | Question | Hypothesis |
| --- | --- | --- |
| RQ1 | How much of the query stream can the router decide at matched risk? | Best router reaches coverage of at least [N]% at matched risk on Banking77 |
| RQ2 | Does pairwise matching beat a softmax classifier at rejecting out-of-scope queries? | Pairwise matcher beats softmax encoder on OOS AUROC by at least [M] points |
| RQ3 | Can new intents be added without retraining? | With 10 support examples per held-out intent, pairwise matcher is within [K] accuracy points of a fully retrained softmax model |
| RQ4 | Does label noise change the conclusions? | System ranking is stable between original and cleaned labels (report if not) |
| RQ5 | Is the router stable under rewording? | Prediction flip rate under validated paraphrases is below [F]% for the best router |
| RQ6 | Does it survive domain shift? | Coverage at matched risk on the Jeff set drops by no more than [D] points versus Banking77 |

A hypothesis that fails is a result, not a problem. Report it plainly.

---

## 5. Non-goals

- No dialogue management, slot filling or response generation.
- No emotion, sentiment, hate speech or sarcasm components.
- No training a language model from scratch.
- No real customer data, bank data or CDR data in this repository, ever.
- No product-grade UI. The demo is worth 20 points, so it is not optional and not last: a thin version exists by the end of M2 and is polished in M6 (section 12.2). It stays a single screen in a single file.
- No leaderboard chasing on Banking77. A plain, well-evaluated system beats a tuned one.

---

## 6. Data

Verify every dataset ID and split size on the Hugging Face Hub before use. IDs have moved in the past. Record the exact ID, config and revision hash used.

| Dataset | Role | Key facts |
| --- | --- | --- |
| **Banking77** (Casanueva et al., 2020) | Primary in-scope benchmark | 77 fine-grained banking intents, 13,083 queries, 10,003 train and 3,080 test. No official validation split: carve a stratified 10% validation set from train with a fixed seed. |
| **CLINC150** (Larson et al., 2019) | Out-of-scope queries, second benchmark | 150 intents over 10 domains (one is banking). 15,000 train, 3,000 validation, 4,500 test in scope. Out of scope in the Hub config `plus` used here: 250 train, 100 validation, 1,000 test (the paper's "full" variant has 100 OOS train). |
| **Banking77 cleaned** | Label-noise experiment | Built in this project with cleanlab (confident learning). Keep the list of flagged indices in `data/processed/banking77_flagged.csv`. Hand-check a random 100 flagged items and report how many are real errors. **Built 22 Nov 2026 (M4b):** 192 of the 9,992 train and validation rows flagged (163 train, 29 validation) by `scripts/find_label_issues.py`; flagged rows are removed, never relabelled (`data.load_train("cleaned")`). Test was not checked. Blind audit sheet: `results/e2_audit.csv`. **Audited 23 Nov 2026:** 61 of 100 flagged items are real label errors (95% Wilson interval 51 to 70%), 10 ambiguous, 29 false flags; the owner judged items 1 to 11 and a Claude judge the rest (section 8.1). |
| **Jeff set** (written by a Claude agent at the owner's request) | Domain-shift test | See 6.1. |

Optional, only if time allows: Banking77-OOS (verify that it exists and how it was built before relying on it).

**Pinned versions and licences** (checked 22 Nov 2026, recorded in `configs/data.yaml`):

- `PolyAI/banking77`, revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, CC-BY-4.0. Banking77 by Casanueva et al. (2020), PolyAI.
- `clinc/clinc_oos`, config `plus`, revision `155b9c710419136e17307b80d0a13e68cd46b4ec`, CC-BY-3.0. CLINC150 by Larson et al. (2019), Clinc Inc.

Both licences allow redistribution with attribution, so the processed splits in `data/processed/` are committed to this repo. The attribution above covers them.

### 6.1 The Jeff set

Planned as owner-written. **Changed 23 Nov 2026 at the owner's request:** written by a separate Claude agent (Opus 5.5) from the intent definitions in `configs/jeff.yaml`. Those come from Jeff's tool belt in the owner's Velm app: 12 tools, plus the two answers Jeff gives from its state block (safe-to-spend and bills due soon). The agent saw no project code, data, models or results. It is still the project's own data contribution, but model-written, not user-written (limitation below).

- 8 to 15 intents that map to the assistant's real tools.
- 20 queries per intent, phrased the way a real Australian user would type them (PayID, BSB, super and similar terms are expected).
- 40 out-of-scope queries, a mix of clearly unrelated and finance-adjacent but unsupported.
- Split per intent: 10 **support** examples (the few-shot examples the matcher is allowed to see) and 10 **test** examples. Out-of-scope queries are test only.
- File: `data/jeff/jeff_queries.csv` with columns `id, text, intent, split, notes`. Use `oos` as the out-of-scope label; its `notes` say `unrelated` or `finance-adjacent`. The writer's file is `data/jeff/jeff_written.csv`; `scripts/make_jeff_set.py` checks it and assigns the splits at random (seed 0), so no one picks which examples go to support.
- **Never train or tune thresholds on the Jeff test split.** It is run once, at the end.

Status: **built 23 Nov 2026.** 14 intents × 20 queries plus 40 out-of-scope (15 unrelated, 25 finance-adjacent): 320 queries, 140 support and 180 test. No support-test pair within an intent is a near-duplicate. A second Claude agent labelled every query blind and agreed with the writer on 320 of 320, marking 6 as unsure (`results/jeff_label_check.md`). Limitation: the set is model-written, so it is probably cleaner than real traffic, and S4 is a Claude model too, which may make it easier for S4 than real users' messages would be.

### 6.2 Preprocessing pipeline

The rubric marks "preprocessing pipeline quality" directly. All preprocessing lives in `src/router/preprocess.py` as plain functions, is called by every system, and is covered by unit tests. Steps, in order:

1. **Load and pin.** Load each dataset by exact Hub ID and revision hash. Save the label-to-ID mapping to `data/processed/label_map.json` so every system uses the same one.
2. **Normalise text.** Unicode NFKC normalisation, collapse repeated whitespace, strip leading and trailing whitespace. Keep case, punctuation and digits: the transformer tokenisers handle them, and amounts and account terms carry meaning. Lowercasing happens only inside the S0 TF-IDF vectoriser.
3. **Drop empty rows and exact duplicates** within each split. Log how many were removed.
4. **Leakage check.** After normalisation, find any text that appears in both train and test. Report the count, remove those rows from train (never from test), and save the list.
5. **Split.** Stratified 10% validation split from train with a fixed seed, saved to disk once (rigour rule 6).
6. **Length check.** Compute token-length statistics per tokeniser on train. Set `max_length` from them and report the share of queries that get truncated.
7. **Class balance.** Report queries per intent for each split. If intents are uneven, that is the reason macro averages are the primary metric (section 8.1).
8. **Out-of-scope mixing.** When CLINC150 out-of-scope queries are added to Banking77, they pass through the same steps 2 and 3. The mixing ratio is set in the config and stated in the report.
9. **Fit on train only.** Anything learned from data (TF-IDF vocabulary, temperature, thresholds) is fitted on train or validation and never on test.

Output: `results/data_report.md`, written by `scripts/make_splits.py`, with split sizes, rows removed at each step, leakage count, length statistics and class balance. Two or three of these numbers go in the summary report as the evidence for this rubric line.

Keep it this simple. No stemming, stop-word removal or spelling correction unless a validation result shows it helps, in which case record it in the decision log.

---

## 7. Systems

All systems output, per query: a predicted intent, an in-scope confidence score and an out-of-scope score. That shared interface is what makes the comparison fair.

| ID | System | Notes |
| --- | --- | --- |
| S0 | TF-IDF plus logistic regression | Classical baseline. Confidence is max predicted probability. |
| S1 | Fine-tuned encoder with softmax head | A small encoder (for example DistilRoBERTa or DeBERTa-v3-small). Confidence is max softmax probability, temperature-scaled on validation. |
| S2 | Embedding nearest neighbour | Frozen sentence-embedding model, cosine similarity to labelled examples. No training. |
| S3 | Pairwise paraphrase matcher | Cross-encoder scoring (query, example) pairs: "do these mean the same thing?". Initialise from an NLI or paraphrase checkpoint. Use S2 to retrieve the top-k candidates so inference stays fast. This is the topic 2 component and the main system. |
| S4 | LLM zero-shot | Given the intent list and descriptions. Use the same model family the assistant runs in production (owner to confirm exact model). Temperature 0. Cache every response to disk. This is the accuracy ceiling and the cost reference, not a competitor to beat. **Confirmed 22 Nov 2026:** `au.anthropic.claude-sonnet-4-6` on Amazon Bedrock (exactly Jeff's production model), zero-shot with the 77 intent names (Banking77 has no descriptions) plus `out_of_scope`, temperature 0, structured JSON answer, prompt in `prompts/s4_intent_v1.txt`, every reply cached in `results/s4_cache/` (`src/router/systems/s4_llm.py`, `scripts/run_s4.py`). |

### 7.1 Router policy

- **Policy v1 (required):** HANDLE if confidence is at or above `t_high`, otherwise ESCALATE.
- **Policy v2 (stretch):** add REJECT when the out-of-scope score is at or above `t_oos`.

Thresholds are chosen on validation data only. The dangerous error is a confident wrong HANDLE, above all on an out-of-scope query. Report that error separately.

### 7.2 Why these models

The rubric marks "appropriateness of chosen model". Each system is in the comparison for a stated reason, and the summary report gives that reason in one line each.

| System | Why it is here |
| --- | --- |
| S0 | The cheapest possible baseline. Shows how much of the task is solved by keywords alone. If S0 is close to S1, say so. |
| S1 | The standard approach for this topic and the reference point a marker expects to see. |
| S2 | Needs no training and takes a new intent by adding examples. Also serves as the retriever for S3. |
| S3 | The main system. Fits requirement R5 (new intents without retraining), and because it scores pairs, every decision can be shown next to the example it matched. That is what the demo uses to explain why it works. |
| S4 | The thing being replaced. Sets the error rate to match and the cost to beat. |

Small encoders are chosen because of the latency requirement (R4) and the compute limits in section 14. The choice between candidate checkpoints is made on validation accuracy and measured CPU latency, and is recorded in the decision log. The "best router" for the headline result is picked by a rule fixed before any test run: highest coverage at matched risk on validation.

---

## 8. Experiments

| ID | Experiment | Data | Reports |
| --- | --- | --- | --- |
| E1 | In-scope accuracy, all systems | Banking77 test | Accuracy, precision, recall and F1 (macro and weighted), per-intent precision, recall and F1 for the 10 worst intents, the most confused intent pairs from the confusion matrix |
| E2 | Label noise | Banking77 original vs cleaned | E1 metrics on both, change in system ranking, audit of 100 flagged items |
| E3 | Out-of-scope rejection | Banking77 test plus CLINC150 out-of-scope test; CLINC150 as its own benchmark | AUROC, FPR at 95% TPR, OOS recall at the chosen threshold |
| E4 | New intents without retraining | Hold out 10 Banking77 intents, add back with 5 and 10 support examples | Accuracy on held-out intents, accuracy on the original 67, versus a fully retrained S1 |
| E5 | Paraphrase robustness | Reworded Banking77 test queries | Flip rate, accuracy drop. Hand-validate 100 paraphrases for meaning preservation and report the pass rate. |
| E6 | **Headline: risk-coverage and cost** | Banking77 test with OOS mixed in | Risk-coverage curves, AURC, coverage at matched risk, LLM calls avoided, cost per 1,000 queries, latency p50 and p95 |
| E7 | Domain shift | Jeff test split | E6 metrics, run once |

### 8.1 Metric definitions

The four metrics the rubric names for classification come first, in E1 and in the summary table:

- **Accuracy:** share of in-scope test queries given the correct intent.
- **Precision, recall and F1:** computed per intent, then averaged. Macro averages are primary because each of the 77 intents should count equally. Weighted averages are reported next to them. Use scikit-learn (`precision_recall_fscore_support`, `classification_report`), not hand-written code.
- **S4 is scored the same way.** It is an LLM, but here it outputs an intent label, so classification metrics apply and generation metrics do not.

The router metrics are then defined in the same vocabulary, so a marker can see they are the same ideas applied to the HANDLE decision:

- **HANDLE precision:** of the queries the router handled, the share it handled correctly. This equals 1 minus selective risk.
- **HANDLE recall:** of the queries whose top prediction was correct, the share the router did handle instead of escalating.
- **Out-of-scope precision and recall:** for the out-of-scope decision at the chosen threshold, alongside AUROC.

Out-of-scope detection (E3), computed by `scripts/evaluate_oos.py` from each system's `oos_score` (1 minus confidence):

- **AUROC:** OOS is the positive class. The chance that a random OOS query scores higher than a random in-scope one; 1.0 is perfect, 0.5 is chance. scikit-learn `roc_auc_score`.
- **FPR at 95% TPR** (in-scope as the positive class, the Hendrycks and Gimpel convention): the threshold is the lowest `oos_score` that keeps at least 95% of in-scope queries (a query is kept when its score is at or below it). FPR is the share of OOS queries also kept. Lower is better.
- **OOS recall at the chosen threshold:** the threshold is the one from the 95% rule, set on validation per seed. On validation, OOS recall is 1 minus the FPR by construction. On test, that threshold is applied unchanged, so test OOS recall is out of sample.
- **Format control:** CLINC150 queries are all lowercase and unpunctuated, unlike most Banking77 queries, so formatting alone separates them (validation: "starts lowercase" scores AUROC 0.971). The primary E3 numbers therefore use format-controlled text: every evaluated query, in-scope and OOS, is lowercased and stripped of punctuation (apostrophes kept) by `router.preprocess.format_control` before prediction (`scripts/predict_format_control.py`, splits ending in `_fmt`). Models and support examples are unchanged. Raw-text numbers are reported next to them.
- **Comparisons:** paired, stratified bootstrap, 1,000 resamples. In-scope and OOS queries are resampled separately (same class counts every time), both systems use the same rows, and each system's AUROC is the mean over its seeds. Report the difference with its 95% percentile interval. The function `paired_bootstrap_diff` in `src/router/metrics.py` is reused for later headline comparisons.

Selective classification terms:

- **Coverage:** share of queries the router decides itself (HANDLE or REJECT).
- **Selective risk:** error rate on the covered queries only.
- **Coverage at matched risk:** highest coverage where selective risk is at or below S4's error rate on the same test set. If S4 cannot be run (no budget or no approval), report coverage at fixed risk levels of 1%, 2% and 5% instead, and say so.
- **AURC:** area under the risk-coverage curve. Lower is better.
- **Flip rate:** share of queries whose predicted intent changes after a validated paraphrase.
- **Cost model:** `cost_per_query = router_cost + (1 - coverage) * llm_cost`. Compute `llm_cost` from measured input and output tokens times a per-token price supplied by the owner. Do not hardcode a price from memory. State the price and its date in the report.
- **Latency:** wall-clock per query on one named machine, batch size 1, warm model.

E6 as implemented (`scripts/evaluate_router.py`, fixed 22 Nov 2026 before E6 was run):

- **Correct:** HANDLE is correct when the predicted intent equals the true label. S0 to S3 cannot answer "out of scope", so handling an out-of-scope query is always an error; it is counted separately as "OOS handled".
- **Matched risk r\*:** S4's error rate on the same mixed set (in-scope plus out-of-scope, format-controlled text). Per system and seed, the threshold is the one with the highest coverage whose selective risk is at or below r\*. Coverage at fixed risk 1%, 2% and 5% is reported too.
- **AURC:** the step-function area under the risk-coverage curve (each distinct threshold's risk times the coverage it adds).
- **Calibration:** S0 and S1 are temperature-scaled before routing (one temperature per seed, grid search minimising NLL on validation in-scope rows); ECE with 15 bins is reported before and after. S2 (cosine) and S3 (one sigmoid) are not rescaled, because a monotone rescaling cannot change their routing.
- **Prices:** `llm_cost` is S4's measured mean cost per call (token usage, including prompt-cache reads and writes, times the Bedrock prices in `configs/s4_llm.yaml`: AWS Price List API, ap-southeast-2, regional `au.` profile, checked 22 Nov 2026). `router_cost` is taken as US$0 (a local model on hardware the assistant already runs); its time cost is reported as latency.
- **Frozen for test:** per-seed temperatures and thresholds, and the best router (highest mean coverage at matched risk), are written to `results/policy_v1.json` and applied unchanged at M5.

E2 as implemented (`scripts/find_label_issues.py`, `scripts/evaluate_label_noise.py`, `configs/e2_label_noise.yaml`, fixed 22 Nov 2026 before the cleaner ran):

- **Rows checked:** all in-scope train and validation rows, pooled (9,992). Test is not checked.
- **Out-of-sample probabilities:** frozen `all-mpnet-base-v2` embeddings (not a compared system: S2 chose MiniLM) and logistic regression, 5-fold stratified cross-validation, seed 0. C from {0.1, 1, 10, 100} by the lowest cross-validated log loss. Saved in `data/processed/banking77_cleaner_pred_probs.npz`, so the flags can be re-derived in a second.
- **Flags:** `cleanlab.filter.find_label_issues` with its defaults (`prune_by_noise_rate`), ranked by self-confidence. The suggested label is the most probable label other than the given one.
- **Cleaned:** flagged rows removed, never relabelled. Every system is retrained on cleaned train with every other choice frozen (S0 keeps its grid parameters, S1 and S3 their hyperparameters, five seeds each; S2 rebuilds its support set). Runs are saved as `<tag>_cleaned` and never appear in E1, E3 or E6.
- **Stop rule before retraining:** more than 10% of train flagged, any intent losing more than a third of its train rows, or cleaner accuracy below 85%.
- **Metrics:** in-scope macro-F1 and accuracy in four cells: train labels {original, cleaned} by validation rows {all, flagged removed}. **Ranking stable (RQ4)** means the same order by mean macro-F1 in all four cells. Training effect per system: cleaned minus original macro-F1, paired bootstrap stratified by intent, 1,000 resamples. S4 is not retrained; its mixed-set error rate (the E6 matched risk) is reported with and without the flagged rows, from cached replies only.
- **Audit:** 100 flagged rows drawn at random (seed 0), shown blind: the given and suggested labels appear as options A and B in random order, each with 3 unflagged train examples. The key is in a separate file. Scoring, fixed before the audit: a real error is the suggestion or "neither"; "both" is ambiguous; the given label is a false flag. Reported with a Wilson 95% interval.
- **Who judged (changed 23 Nov 2026 at the owner's request):** the owner judged items 1 to 11; a Claude judge (Opus 5.5, a separate Claude Code subagent) judged all 100 items from a copy of the sheet without the verdict and notes columns. It never saw the key, the flagged list or the owner's verdicts. The sheet's `annotator` column says whose verdict counts for each item (the owner's for 1 to 11, the judge's for 12 to 100); the judge's verdicts on all 100 are in `results/e2_audit_claude.csv`. Results are reported per annotator and together, plus owner-judge agreement on items 1 to 11. Limitation: a model judge may remember public Banking77 labels, which would push it towards the given label (fewer real errors).

---

## 9. Rigour rules (non-negotiable)

1. **Test sets are touched once per system**, after all choices are frozen. All tuning happens on validation.
2. **Five random seeds** for every trained system. Report mean and standard deviation.
3. **Paired bootstrap confidence intervals** (1,000 resamples) for every headline comparison.
4. **No number goes in the report unless a script in this repo produced it.** Never estimate, round up or fill in a result that was not run.
5. **Every result file records** the git commit, config, seed, dataset revision and timestamp.
6. **Same splits for every system.** Splits are created once by `scripts/make_splits.py` and saved.
7. **LLM baseline is reproducible:** prompt stored in the repo, temperature 0, responses cached, model ID and date recorded.
8. **Error analysis is done by reading examples**, not only by metrics. At least 100 router errors are read and categorised by hand.
9. **Negative results are reported.**
10. **The submission notebook passes a clean-run check** before submission (section 10.1). After M1 it must run end to end at the close of every working session.

---

## 10. Repository layout

```
.
├── README.md                  this file
├── pyproject.toml             dependencies, pinned
├── configs/                   one YAML per system and experiment
├── data/
│   ├── raw/                   downloaded datasets (gitignored)
│   ├── processed/             splits, cleaned labels, paraphrases
│   └── jeff/                  the Jeff set (intents defined in configs/jeff.yaml)
├── src/router/
│   ├── data.py                loading, splits
│   ├── preprocess.py          text normalisation, dedup, leakage check (section 6.2)
│   ├── systems/               s0_tfidf.py ... s4_llm.py, shared interface
│   ├── policy.py              HANDLE / REJECT / ESCALATE logic
│   ├── calibration.py         temperature scaling
│   ├── metrics.py             risk-coverage, AURC, OOS metrics, bootstrap
│   ├── label_noise.py         cleanlab flags, cleaned splits, blind audit (E2)
│   ├── jeff.py                Jeff set checks, support/test split, label agreement
│   └── cost.py                cost and latency model
├── scripts/                   make_splits, clean_labels, train, evaluate, make_figures
├── results/                   metrics as JSON or CSV, figures, preregistration.md, data_report.md, clean_run.md
├── notebooks/
│   ├── 00_submission.ipynb    the notebook the marker runs (section 10.1)
│   └── ...                    exploration and error analysis
├── demo/
│   ├── app.py                 single-file demo (section 12.2)
│   └── DEMO_SCRIPT.md         the queries and talking points for the demo
├── report/
│   ├── summary.pdf            the 1 to 2 page summary that is submitted (section 12.1)
│   └── appendix.md            full method and results, linked from the summary
└── tests/                     unit tests for preprocessing, metrics and policy
```

### 10.1 The submission notebook

`notebooks/00_submission.ipynb` is what the marker runs. The rubric line is "notebook runs end-to-end without breaking", so it is a deliverable with its own rules:

- **It is thin.** It imports from `src/router/` and calls functions. No logic lives only in the notebook, so the modular structure is visible and the notebook cannot drift from the scripts.
- **It follows the rubric order:** problem, data and preprocessing, models, evaluation (accuracy, precision, recall, F1 first), router and cost, limitations. Each part opens with two or three sentences of markdown.
- **`FAST = True` by default.** In fast mode it trains S0 from scratch, loads saved checkpoints for S1 and S3 (or trains for a small fixed number of steps if none is found), and reads S4 responses from the disk cache. It needs no API key, no GPU and no paid call. Set a run-time target, measure it on the free Colab tier, and record the measured time.
- **`FAST = False`** reproduces the full five-seed results.
- **Fast-mode numbers are labelled as fast-mode numbers.** The reported results come from the full-run result files and are loaded and displayed next to them (rigour rule 4).
- The first cell installs the pinned dependencies and prints versions. The second sets the seeds.
- **Clean-run check:** fresh clone, fresh environment, Restart and Run All, on two machines, one of them Colab. Record the date, commit and run time in `results/clean_run.md`.

Open point for the owner: where the S1 and S3 checkpoints and the S4 cache are hosted so the notebook can fetch them (in the repo if small enough, otherwise a Hub model repo), and whether the dataset licences allow committing cached outputs.

Suggested stack: Python 3.12 (managed with `uv`; see decision log), PyTorch, Hugging Face `transformers` and `datasets`, `sentence-transformers`, scikit-learn, cleanlab, pandas, matplotlib. Check that each model checkpoint name exists on the Hub before using it.

---

## 11. Milestones

No deadline has been supplied yet, so these are ordered phases. Convert to dates once the owner provides one.

| Phase | Work | Done when |
| --- | --- | --- |
| M0 | Repo scaffold, environment, data download, preprocessing (section 6.2), fixed splits, empty submission notebook that imports the package | `make_splits.py` runs clean, split sizes match section 6, `results/data_report.md` is written, the notebook runs top to bottom |
| M1 | S0 and S1 on Banking77, validation only. E1 metric code (accuracy, precision, recall, F1). Skeleton of the two-page summary with an empty results table. | Validation accuracy in a sane range, seeds reproducible, the notebook runs end to end with S0 and S1. **This is the minimum submittable version. Keep it working from here on.** |
| M2 | S2 and S3, shared scoring interface. Thin demo. | All four local systems emit the same output schema. The demo answers a typed query offline. |
| M3 | Out-of-scope evaluation (E3) on validation, with the format control (section 8.1). Reordered 22 Nov 2026: E2 moved to M4b | OOS metrics on validation with bootstrap comparisons, notebook section, tests |
| M4 | Metrics module with tests, calibration, router policy, S4 baseline | Risk-coverage code unit-tested on toy data, LLM responses cached |
| M4b | Label cleaning (E2), started only once every core-path item in section 0 works (needs cleanlab, approved 22 Nov 2026) | Flagged list saved, 100 items audited (changed 23 Nov 2026: the owner judged 11, a blind Claude judge the other 89), E1 on original vs cleaned labels. **Done 23 Nov 2026** (`results/e2_val.md`) |
| M5 | Freeze everything, run E1 to E7 on test | `results/` complete, preregistration file predates the test runs |
| M6 | Error analysis, figures, two-page summary, appendix, demo polish and script, clean-run check | Summary fits two pages and is reviewed by the owner, clean run recorded, demo rehearsed against a timer |

Cut order if time runs short: policy v2 (REJECT), then E5, then CLINC150 as a second full benchmark, then E4, then E2. Never cut the submission notebook, the demo, the two-page summary, E1 with precision, recall and F1, E6, the seeds or the confidence intervals. Those are where the 100 points are.

---

## 12. Deliverables

1. Code in this repo, runnable end to end from a clean clone, with the submission notebook (section 10.1).
2. A 1 to 2 page summary report (section 12.1). Two pages is a hard limit set by the rubric.
3. One headline figure (risk-coverage with cost axis) and one summary table, both inside the two pages.
4. A working demo application (section 12.2).
5. `results/preregistration.md` and the Jeff set.
6. `report/appendix.md`: related work, full method, all experiment tables, error analysis and deployment notes. Linked from the summary. Do not rely on it being read.

### 12.1 The two-page summary

Laid out so each rubric category can be found without searching. Build the skeleton in M1 with an empty table to find out what fits.

| Part | Space | Content | Rubric category |
| --- | --- | --- | --- |
| Problem and constraints | About a quarter page | The task, the user, the budget and latency constraints, the one research question | 1 |
| Data and preprocessing | About a quarter page | Datasets and why, split sizes, two or three numbers from `results/data_report.md` | 2 |
| Systems | About a quarter page | One line per system saying why it is there (section 7.2) | 2 |
| Results | About three quarters of a page | One table: accuracy, macro precision, recall and F1 per system, mean and standard deviation. The headline figure. Coverage at matched risk and cost saved in one sentence. Requirements R1 to R7 marked met or not met. | 4, 1 |
| What it does, why it works, limitations | About a quarter page | Same three headings the demo uses. Limitations are specific: out-of-scope data from a different source, small model-written Jeff set, label noise, single domain. | 5 |
| Footer | A few lines | Repository link, how to run the notebook, AI use disclosure as the unit requires, key references | 3 |

Stretch results get one sentence or one table row each. Anything that does not fit goes to the appendix.

### 12.2 The demo

Worth 20 points. The rubric asks for a clear, engaging demonstration that explains what it does, why it works and its limitations. The demo is designed around those three things.

- **Form (decided 22 Nov 2026):** the real Jeff chat screen from the owner's app, run as a dev-only Flutter harness (`apps/mobile/lib/jeff_router_demo_main.dart` in the Velm repo, a new file, nothing existing changed), with the router in front of it. `demo/app.py` in this repo is the router: a local server (Python standard library only, no new dependency) that calls the same `src/router/` code as the notebook. Run `uv run python demo/app.py` here, then `flutter run -t lib/jeff_router_demo_main.dart -d chrome` in `Velm-Main/apps/mobile`. A side panel holds the threshold slider, the session counts and the evidence for the last decision. The Velm harness is not part of the submission; the marker sees it live.
- **What it does:** type a query, see the decision (HANDLE, ESCALATE or REJECT), the predicted intent, and the confidence next to the threshold.
- **Why it works:** show the closest matched support example and its pair score, so the viewer sees what the router compared the query to. A threshold slider shows the trade between coverage and risk live.
- **What it saves:** a running count of LLM calls avoided and estimated cost saved in the session.
- **Limitations:** shown, not just stated. The script includes one query the router gets confidently wrong, taken from the error analysis.
- **It runs offline on a laptop CPU.** An escalated query is labelled "would be sent to the LLM". No live API call on the day.
- **Scripted run:** `demo/DEMO_SCRIPT.md`, about three minutes, six queries in this order: a clear in-scope query, the same request reworded, a query that sits between two intents (escalates), a clearly unrelated query, a finance-adjacent but unsupported query, and the known failure. Then the limitations in three sentences.
- **Backup:** record a screen capture of the scripted run in case the live demo fails.

---

## 13. Working agreement for Claude Code

**About the owner**

- Krishil is a Bachelor of Data Science student. This is an individually assessed project, so he must be able to explain and defend every design choice and every line of code.
- He does not retain Python syntax well. Keep code simple, typed and commented. Prefer plain functions over clever abstractions. When introducing a non-obvious construct, add a one-line comment on what it does.
- He wants direct, critical feedback. Lead with the problem, then the fix. Do not soften bad news and do not agree with a weak idea to be polite.

**How to work**

- Start each session by reading this README and `results/` to see what exists.
- Work one milestone at a time. State what you are about to do, do it, then report what ran and what the numbers were.
- Ask before: any run expected to take more than about 30 minutes, any paid API call, any new dependency, any change to the research questions.
- Never fabricate, estimate or "expect" a result. If something was not run, say so.
- If a result looks too good, assume a leak or a bug first and check splits before celebrating.
- Flag scope creep. If the owner asks for a new component, ask which research question it serves.
- Before calling a milestone done, check it against section 0: which rubric category did this work earn points in, and where will the marker see it?
- Do not start stretch work while any core-path item in section 0 is missing or broken.
- Update this README's status line and the decision log when something material changes.

**Writing style for the report and all prose**

- Plain, direct English. No em dashes and no other stock AI phrasing.
- Claims match evidence. No "state of the art", no "novel framework".
- Cite every dataset and method.

**Academic integrity**

- Confirmed 22 Nov 2026: the unit allows AI-assisted code with disclosure. Every session that Claude Code writes code is logged in `report/ai_use_log.md`, which is the source for the disclosure. Whatever the policy allows, the owner writes or rewrites the final report in his own words and discloses AI use as the unit requires.

---

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| The plan is far larger than a two-page summary can show, so depth goes unmarked | Core path first (section 0). Every experiment ends as one row, one figure or one sentence. Page budget in section 12.1. |
| The notebook breaks on the marker's machine (no GPU, no API key, different versions) | Fast mode with cached LLM responses, pinned dependencies, clean-run check on two machines (section 10.1). |
| The marker looks for accuracy, precision, recall and F1 and finds only risk-coverage curves | E1 table comes first in the notebook and the summary. Router metrics are defined in the same terms (section 8.1). |
| The live demo fails on the day | Runs offline with no API call, scripted, rehearsed, with a recorded backup (section 12.2). |
| Lecturer does not accept topic 7 with a custom domain and a topic 2 method | Get written confirmation first. Fallback is Plan B in section 16. |
| Banking77 and the assistant's real intents barely overlap | Expected. That is what E7 measures. Do not force a mapping. |
| Pairwise matcher is too slow | Retrieve top-k with S2, score only those pairs. Report latency as a result. |
| S4 is near perfect and makes the router look pointless | The argument is cost and latency at matched risk, not accuracy. If coverage at matched risk is low, report it. |
| Out-of-scope data comes from a different dataset than in-scope data, so rejection looks easier than it is | State this limitation. The Jeff set's finance-adjacent OOS queries are the harder test. |
| Paraphrases change meaning | Hand-validate 100 and report the pass rate. Drop failed ones. |
| Compute limits | All local models are small. Everything should run on a single consumer GPU or a free Colab tier. Confirm with the owner. |

---

## 15. Open items (owner input needed)

- [ ] Submission deadline and any interim checkpoints
- [x] Marking rubric and required report format or length (received 22 Nov 2026: 100 points over five categories, 1 to 2 page summary plus code, brief demo, see section 0)
- [ ] Demo format: live or recorded, how long, in person or online, and when
- [ ] Whether an appendix or extra pages beyond the two-page summary are allowed or read
- [x] Whether the code must be a single notebook or a repository with a notebook is accepted (22 Nov 2026: a repository with a thin notebook is accepted)
- [ ] Where model checkpoints and the S4 response cache are hosted for the submission notebook (dataset licences checked 22 Nov 2026: both CC-BY, so committing processed data and cached outputs is allowed with attribution; hosting for checkpoints still open)
- [ ] A GitHub repository for the code (the Colab clean-run check clones it, and the summary footer links it)
- [x] Approval for Gradio as a demo dependency (22 Nov 2026: not needed; the demo uses the real Jeff UI from the Velm app with a standard-library router server, section 12.2)
- [ ] Latency limit L in requirement R4 (section 2.5)
- [ ] Written lecturer confirmation that this topic framing is accepted
- [x] The unit's policy on AI assistance (22 Nov 2026: allowed with disclosure, see section 13)
- [ ] Available compute (local GPU, Colab, other). Measured 22 Nov 2026: the owner's Apple M3 Pro (18 GB, MPS) trains one S1 DistilRoBERTa seed in about 5.5 minutes. It cannot train DeBERTa-v3-small in float32 (memory), so larger models need Colab or another GPU
- [x] The assistant's intent list (8 to 15 intents) and the Jeff set (23 Nov 2026: 14 intents from Jeff's tool belt in `configs/jeff.yaml`; the set was written by a Claude agent at the owner's request, section 6.1)
- [x] Exact LLM model for S4, its per-token price and a spend cap for the baseline runs (22 Nov 2026: `au.anthropic.claude-sonnet-4-6` on Bedrock, profile `unibucks`; US$3.30 / US$16.50 per million input / output tokens from the AWS Price List API; spend cap US$10 for M4, validation only. Test and Jeff-set runs at M5 need a fresh approval)
- [ ] Threshold values N, M, K, F, D in section 4 (drafted 23 Nov 2026 by Claude at the owner's request, with L, in `results/preregistration_draft.md`; waiting for the owner's confirmation)

---

## 16. Plan B

If the topic framing is rejected, switch to **topic 6 plus topic 2: "Can ROUGE tell a paraphrase from a lie?"** Take good summaries, create meaning-preserving paraphrases and factual corruptions (swapped number, swapped entity, added negation), and test whether ROUGE, BLEU and BERTScore separate the two. Valuable to anyone shipping LLM summaries. Less deployable, which is why it is second.

---

## 17. Key references

- Casanueva et al. (2020). Efficient Intent Detection with Dual Sentence Encoders. (Banking77)
- Larson et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP. (CLINC150) https://aclanthology.org/D19-1131
- Ying and Thomas (2022). Label Errors in BANKING77. Insights workshop, ACL. https://aclanthology.org/2022.insights-1.19
- Zhang et al. (2020). Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference. EMNLP. https://arxiv.org/abs/2010.13009
- Northcutt et al. (2021). Confident Learning: Estimating Uncertainty in Dataset Labels. (cleanlab)
- Guo et al. (2017). On Calibration of Modern Neural Networks. (temperature scaling)
- Geifman and El-Yaniv (2017). Selective Classification for Deep Neural Networks. (risk-coverage)
- Hendrycks and Gimpel (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. (max softmax baseline)

Read the first four before M1. Confirm every figure quoted in this README against the source paper before it goes in the report.

---

## 18. Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 13 Nov 2026 | Dropped the hate speech plus emotion project | Over-subscribed, sprawling, central finding already published |
| 13 Nov 2026 | Chose topic 7 with topic 2 as the method | Real use case, measurable cost result, clean evaluation |
| 13 Nov 2026 | Excluded emotion detection from scope | Would repeat the sprawl problem |
| 13 Nov 2026 | Headline metric set to coverage at matched risk | Ties the classifier result directly to LLM cost |
| 22 Nov 2026 | Added section 0 mapping the project to the marking rubric, with a core path and a stretch layer | Rubric received. Marks sit in a notebook, a demo and two pages, not in a long report |
| 22 Nov 2026 | Report cut from a full report to a 1 to 2 page summary plus an appendix | Rubric submission format |
| 22 Nov 2026 | Demo moved from "last task, first to cut" to a core deliverable with a thin version at M2 | Demo and communication is 20 of 100 points |
| 22 Nov 2026 | Added a submission notebook with a fast mode and a clean-run check | Rubric line "notebook runs end-to-end without breaking" |
| 22 Nov 2026 | Added precision and recall to E1 and defined the router metrics in those terms | Rubric names accuracy, F1, precision and recall for classification |
| 22 Nov 2026 | Wrote down the preprocessing pipeline, the model rationale and the requirements table | Rubric lines on preprocessing quality, model appropriateness and requirement coverage |
| 22 Nov 2026 | Python 3.12 managed by `uv`, exact `==` pins in `pyproject.toml`, `uv.lock` committed, `requirements.txt` exported for Colab | 3.12 was already installed and every package has wheels for it; avoids a second interpreter on a nearly full disk |
| 22 Nov 2026 | Both datasets pinned by Hub revision hash in `configs/data.yaml` | `PolyAI/banking77` was modified on 28 Oct 2026, so an unpinned load could change under us |
| 22 Nov 2026 | CLINC150 loaded from Hub config `plus` (250 OOS train, not 100) | The paper's "full" variant is not on the Hub under that name. Only the OOS rows are used, so the difference is just pool size |
| 22 Nov 2026 | Validation OOS pool = CLINC150 OOS train + validation; test OOS pool = CLINC150 OOS test | Gives a similar OOS share in validation and test (about 25%), so thresholds tuned on validation transfer. Legitimate only because no system trains on OOS data. If that changes, this pool must be split |
| 22 Nov 2026 | Duplicate rule: same text with the same label keeps one copy; same text with different labels drops every copy; OOS rows that match a Banking77 text are dropped from the OOS pool | No system can get a conflicting pair right, and a text cannot be both in scope and out of scope |
| 22 Nov 2026 | Processed splits are committed to git; `make_splits.py` refuses to overwrite them without `--force` | Both licences allow it, the notebook then runs without a download step, and the splits are created once (rigour rule 6) |
| 22 Nov 2026 | One shared output for every system: predicted intent, confidence = max softmax probability, OOS score = 1 - confidence; raw logits saved | Fair comparison (section 7). Saved logits let M4 fit temperature scaling without retraining. Validation predictions cover the OOS-mixed set so M4 can tune the router from files |
| 22 Nov 2026 | S0 grid on validation macro-F1: n-grams (1,1)/(1,2) x C {1, 10, 100}. Chosen (1,2), C=100 (0.8895) | Best point is on the grid edge, but the top three are within 0.3 points, so the grid was not widened (section 5, no leaderboard chasing). S0 is deterministic: five seeds give identical predictions, std 0 |
| 22 Nov 2026 | S1 trained with a plain PyTorch loop, fixed hyperparameters (lr 5e-5, AdamW wd 0.01, batch 32, 5 epochs, 10% warmup, clip 1.0, max_length 56), last epoch kept | No `accelerate` dependency; every line explainable. Standard values, not tuned |
| 22 Nov 2026 | S1 checkpoint: `distilbert/distilroberta-base`. Rule fixed before any run: higher validation macro-F1 by more than 0.5 points wins, else lower CPU p95 latency; a candidate that cannot train on MPS loses | `microsoft/deberta-v3-small` could not be trained here. transformers 5 loads it in its stored float16, which made AdamW produce NaN after one step (fixed by loading in float32). In float32 it used 12 GB RAM with swap nearly full on the 18 GB M3 Pro and did not finish 60 steps in 12 minutes. Evidence in `results/s1_selection.json` |
| 22 Nov 2026 | All S1 checkpoints load in float32 explicitly | transformers 5 otherwise uses the checkpoint's stored precision |
| 22 Nov 2026 | Keep all five S1 checkpoints (317 MB each, gitignored); seeds [0, 1, 2, 3, 4] | M5 needs per-seed test predictions, and MPS is not bit-for-bit repeatable (same seed twice: max logit difference 2.1e-6, identical predictions), so retraining could shift the validation numbers |
| 22 Nov 2026 | S2 model: `sentence-transformers/all-MiniLM-L6-v2` (validation macro-F1 0.9211 vs 0.9149 for MPNet, gap 0.62 > 0.5; CPU p50 6.1 ms vs 33.2 ms). S2 runs once (no training, deterministic) | Same selection rule as S1, fixed before the run. `scripts/select_checkpoint.py` replaces `select_s1.py` so S1 and S2 use one implementation |
| 22 Nov 2026 | Banking77 near-duplicates: 26.1% of validation and 29.0% of test queries have a train query with cosine >= 0.95 (`results/near_duplicates.md`). Splits kept; reported as a limitation | Found because S2 (no training) matched fine-tuned S1. Owner approved one data-only check of test text (no labels, no predictions). The rates are similar, so validation stays a fair proxy for test |
| 22 Nov 2026 | S3: paraphrase cross-encoder `cross-encoder/quora-distilroberta-base`; S2 shortlists 10 intents x 2 examples (shortlist contains the true intent for 99.7% of validation queries); 1 epoch, lr 2e-5, batch 32, fixed | A paraphrase-detection model fits topic 2 and is DistilRoBERTa-sized, so it trains on this hardware (DeBERTa-v3 did not) |
| 22 Nov 2026 | S3 training pairs come from each query's own S2 shortlist, for positives and negatives alike | The first version (random same-intent positives, most-similar other-intent negatives) learned "looks similar = different intent" and scored 11.8% on validation, below the untrained model (80.5% on 200 queries). Kept in `results/failed_runs/` as a negative result (rigour rule 9) |
| 22 Nov 2026 | Policy v1 in `src/router/policy.py` (HANDLE at confidence >= t_high, else ESCALATE). The demo uses t_high = 0.5 with a slider until M4 picks it on validation | The thin demo needs a decision now. The threshold that goes in the report is chosen in M4, on validation only |
| 22 Nov 2026 | Demo front end is the real Jeff screen (Velm Flutter harness, new file only), router server in `demo/app.py` with no new dependency. Gradio not used | Owner choice. A real product screen is more convincing for the 20-point demo. The harness calls the local server directly, which the app's own rules forbid for product code, so it stays a dev-only entry point |
| 22 Nov 2026 | Notebook FAST mode measured: 16.6 s with saved checkpoints (M3 Pro); 10.4 min on CPU only without checkpoints, where S1 trains 300 steps (75.8% validation accuracy, labelled as fast mode) | Works without a GPU, but slowly. The fix is hosting the seed-0 S1 checkpoint for the notebook to download (open item in section 15), not fewer steps |
| 22 Nov 2026 | Two-page summary written in Markdown and built to PDF with pandoc + XeLaTeX by `scripts/build_summary.py`, which fails above 2 pages and fills the results table from `results/` | Owner choice. No number is typed by hand (rigour rule 4). Word budgets (730 words in six parts) fit with about a third of page 2 spare |
| 22 Nov 2026 | S3 over five seeds on validation: accuracy 90.22 ± 0.26, macro-F1 90.52 ± 0.31. That is below S2 (92.11) and S1 (91.46 ± 0.44). Reported as is; S3 settings stay fixed | No hypothesis says S3 wins on in-scope accuracy (section 4). S3's case is out-of-scope rejection and new intents (RQ2, RQ3), measured in M3 and later. Tuning it until it wins would be leaderboard chasing (section 5) |
| 22 Nov 2026 | S3 seeds 3 and 4 trained locally on MPS, like seeds 0 to 2 (3.9 and 4.0 min). AWS Lambda was checked and not used | Lambda stops every call at 15 min and has no GPU. S3 took a measured 17.8 min per seed on CPU (M3 Pro probe), so it would need save-and-resume code across calls, and the account's EC2 GPU quota is 0. Training locally keeps all five seeds on one device. Owner chose local |
| 22 Nov 2026 | Notebook FAST mode with all four systems measured: 55 s with saved checkpoints (M3 Pro, MPS); 5.2 min on CPU only without checkpoints, where S1 trains 300 steps (75.8% validation accuracy) and S3 is the paraphrase model without fine-tuning (80.5%), both labelled as fast mode | Runs without a GPU or checkpoints. Hosting the seed-0 S1 and S3 checkpoints (open item, section 15) would let the marker run the reported models |
| 22 Nov 2026 | Demo latency, router server alone on MPS with S3: 27 to 184 ms per query over 8 typed queries (median about 42 ms) | Fast enough for a live demo. The first calls are the slowest |
| 22 Nov 2026 | Milestones reordered: E3 now (M3), then the M4 core, then E2 as M4b. cleanlab approved for M4b | Section 0 marks E2 and E3 as stretch and section 13 forbids stretch work while a core item is missing; E6 and policy v1 are core. E3 on validation needed no training, so it went first. Owner choice |
| 22 Nov 2026 | E3 threshold rule: the lowest `oos_score` that keeps at least 95% of in-scope validation queries, per seed, frozen for test. Comparisons by paired, stratified bootstrap (1,000 resamples, seed 0) | Fixed before any test run (section 8.1). Stratifying keeps the 1,000 / 350 class counts in every resample |
| 22 Nov 2026 | E3 primary numbers use format-controlled text (lowercase, punctuation removed, apostrophes kept) for every evaluated query; raw-text numbers reported alongside | All 350 CLINC150 OOS validation queries are lowercase and unpunctuated, against 5.8% and 13.4% of Banking77, so a "starts lowercase" rule alone scores AUROC 0.971 (`results/e3_val.md`). The control removes that cue (AUROC 0.5) without retraining. It changed little: S3 99.70 to 99.72, S1 97.38 to 96.89. Owner choice |
| 22 Nov 2026 | E3 on validation (format-controlled): AUROC S0 94.80, S1 96.89 ± 0.09, S2 MiniLM 99.91, S3 99.72 ± 0.04. S3 minus S1 +2.82 points (95% CI +2.21 to +3.56); S3 minus S2 -0.20 (CI -0.33 to -0.09) | RQ2 holds on validation. S3 does not beat its own retriever at OOS detection; report it. Two caveats: S2's in-scope scores may be helped by the near-duplicates (26.1% of validation; not measured), and CLINC150 OOS is an easier test than real use |
| 22 Nov 2026 | Notebook FAST mode with the E3 section measured: 1.8 min with saved checkpoints (M3 Pro, MPS); 8.4 min on CPU only without checkpoints | E3 scores the 1,350 format-controlled validation queries with all four systems in the notebook itself, which costs about 3 min on CPU. Still under 10 min without a GPU |
| 22 Nov 2026 | S4 = `au.anthropic.claude-sonnet-4-6` on Amazon Bedrock (profile `unibucks`), zero-shot with the 77 intent names, temperature 0 (sent through `extra_body`), structured JSON answer, every reply cached. SDK pinned to `anthropic[bedrock]==1.17.0` in a separate `llm` dependency group | Owner choice: exactly Jeff's production model (Velm `api_stack.py`). Sonnet 4.6 on Bedrock rejects the API version the 2.x SDK sends, so the docs prescribe `<2`. The notebook only reads the committed cache, so the marker needs no AWS account. botocore forces `python-dateutil<3`, so the lock moved it from 4.0.0 to 2.9.0.post0 (pandas only uses it for date parsing, which this project never does) |
| 22 Nov 2026 | Zero-shot S4 on validation (format-controlled): in-scope accuracy 76.3%, 344 of 350 OOS caught, error 18.0% on the mixed set, US$1.07 for 1,350 calls (prompt cache reads made it cheaper than estimated) | Below every router in-scope. Its errors are fine-grained confusions between near-identical intents and some apparent label noise, not a bug (20 read by hand). As the matched risk it would be a lenient bar |
| 22 Nov 2026 | Added a few-shot S4 variant: same model and settings, with 2 train examples per intent in the prompt (seeded, format-controlled). Rule fixed before the few-shot run and before E6: the variant with the lower validation error sets the matched risk and the price of an escalation; both are reported. Both count toward the one US$10 cap | Owner choice. A stronger S4 can only make the router's bar harder, so changing it after seeing the weak zero-shot result does not flatter the router |
| 22 Nov 2026 | Few-shot S4 on validation (format-controlled): in-scope accuracy 84.3%, OOS recall 97.7%, error 12.22% on the mixed set, 0 invalid replies, US$0.00157 per call (US$2.12 for 1,350 calls). It sets the matched risk. Total S4 spend for M4: US$3.19 of the US$10 cap | Still below the trained routers in-scope (89 to 91%), which learn from about 9,000 labelled examples (the few-shot errors have not been read yet). The few-shot run's provenance records `git_dirty: true`: the code was at `ae8d770`, and the only uncommitted files were regenerated E1/E3 result files. Result files are now committed before every run |
| 22 Nov 2026 | Temperature scaling on validation in-scope rows: S0 T = 0.835, S1 mean T = 0.969 (five seeds). ECE before to after: S0 4.95% to 4.09%, S1 3.79% to 3.68% (in-sample). S3 unscaled, ECE 6.00% | Both softmax systems were already close to calibrated; S0 is slightly under-confident (T below 1 sharpens it). The honest after-scaling ECE comes from test at M5 |
| 22 Nov 2026 | E6 on validation (format-controlled), policy v1 at matched risk 12.22%: coverage S0 70.74%, S1 73.38 ± 0.64%, S2 76.59%, S3 76.18 ± 0.25%. Out-of-scope queries handled (of 350): S0 61, S1 54.4, S2 34, S3 31.6. Cost per 1,000 queries: US$1.571 with the LLM only, US$0.368 with S2, US$0.374 with S3. Best router by the fixed rule: S2. S3 minus S1 +2.80 points (95% CI +0.92 to +4.16); S3 minus S2 -0.41 (CI -1.72 to +0.76) | The router handles about three quarters of queries at the LLM's own error rate, so it avoids that share of LLM calls. S3, the main system, is not better than its own retriever here; the pre-registered rule picks S2, and that is reported as it is. Handling out-of-scope queries is the dangerous error and goes in the demo as the deliberate failure case |
| 22 Nov 2026 | Latency, CPU only (Apple M3 Pro, batch size 1, warm, 200 format-controlled validation queries), p50 / p95: S0 0.3 / 0.3 ms, S1 14.4 / 18.9 ms, S2 5.5 / 6.4 ms, S3 95.7 / 159.7 ms. Few-shot S4 API from Sydney: 682 / 831 ms | Every router answers well inside one LLM round trip; S3 is the slowest because it scores 20 pairs per query |
| 22 Nov 2026 | Notebook FAST mode with section 5 (E6) measured: 1.8 min with saved checkpoints (M3 Pro, MPS); 8.3 min on CPU only without checkpoints. It reads S4 from the committed cache and never calls the API | The marker needs no AWS account. Without checkpoints, fast-mode S1 and S3 are much weaker (coverage at matched risk 43.3% and 37.1%), and the notebook labels them as fast mode |
| 22 Nov 2026 | Added `cleanlab==2.9.0` as a main dependency (the notebook re-derives the E2 flags with it, also on Colab). It brought `termcolor==3.3.0`; no existing pin changed | Approved for M4b. Checked by comparing `uv.lock` before and after |
| 22 Nov 2026 | E2 design fixed before the cleaner ran (section 8.1): pooled train and validation checked; out-of-fold probabilities from frozen MPNet embeddings plus logistic regression; cleanlab defaults; flagged rows removed, not relabelled; every other choice frozen for the retraining; a stop rule; a blind audit with fixed scoring | MPNet is not a compared system (S2 chose MiniLM). Removing rather than relabelling avoids imposing a model's label. The blind audit stops the owner leaning towards either the dataset's label or the suggestion |
| 22 Nov 2026 | E2 cleaner: cross-validated accuracy 92.4%. C = 100 kept, the top of the pre-set grid (CV log loss 0.304 at C = 10, 0.268 at C = 100). cleanlab flagged 192 of 9,992 rows: 163 train (1.81%), 29 validation (2.90%). Largest share in one intent: `top_up_reverted`, 9.9% of its train rows. Stop checks passed. A second run gave identical probabilities and flags | The best C may lie above 100, but the grid was fixed before the run, so it was not widened after seeing the result. The top flags read as real errors (for example "Why did my payment get declined?" labelled `declined_transfer`); lower ones are near-synonym intent pairs (`top_up_reverted` vs `top_up_failed`). How many are real errors is the audit's job |
| 22 Nov 2026 | Retrained on cleaned train (8,829 rows), every other choice frozen: S0 and S2 in seconds; S1 five seeds at 4.4 min each and S3 five seeds at 3.9 to 4.4 min each on MPS, 47 min in all, run alone. Every run recorded a clean git tree | Approved with the M4b plan. Checkpoints grew from 3.2 GB to 6.4 GB |
| 22 Nov 2026 | E2 on validation: by macro-F1, S2 > S1 > S3 > S0 in all four cells, so the ranking is stable (RQ4, validation). Training effect, cleaned minus original macro-F1 with 95% CI: S0 -1.19 [-2.22, -0.12], S1 +0.12 [-0.39, +0.65], S2 +0.54 [+0.01, +1.13], S3 +0.79 [+0.16, +1.48] points. Removing the 29 flagged validation rows raises every system by 1.7 to 2.2 points | The two systems that gained use embeddings (S2, and S3 through its S2 retriever), like the cleaner, and the keyword system lost. That may be the cleaner bias named before the run, or S0 simply losing training data; E2 cannot tell the two apart. The changes are small next to the gaps between systems, so the ranking holds |
| 22 Nov 2026 | S4 with the 29 flagged validation rows removed (cached replies, no API call): few-shot error on the mixed set 12.22% to 10.90%, zero-shot 18.00% to 16.81%. 2 of few-shot S4's 154 prompt examples are flagged train rows | About 1.3 points of the 12.22% matched risk falls on flagged queries, so on cleaned labels the matched risk would be stricter. The routers' errors on those rows would go too, so the net effect on coverage is unknown without re-running E6, which is not part of E2. One sentence for the limitations |
| 22 Nov 2026 | Notebook FAST mode with section 4c (E2) measured: 1.7 min with saved checkpoints (M3 Pro, MPS); 8.1 min on CPU only without checkpoints. Section 4c re-derives the 192 flags from the saved probabilities and matches the committed list in both runs | E2 adds about a second: the slow parts (embedding, cross-validation, retraining) are in FAST = False only. Still under 10 min without a GPU |
| 23 Nov 2026 | E2 audit: the owner judged items 1 to 11 and asked Claude to judge the rest. A separate Claude Code subagent (Opus 5.5) judged all 100 items from a copy of the sheet with no verdicts, without the key or the flagged list. Its verdicts count for items 12 to 100; the owner's count for 1 to 11 | Owner's request. Kept blind so the judge cannot lean towards the given label from the key. The judge also did items 1 to 11 so its agreement with the owner can be measured. Reported per annotator, as an AI-assisted annotation (AI use log) |
| 23 Nov 2026 | E2 audit result: 61 of 100 flagged items are real label errors (95% Wilson interval 51 to 70%; 57 where the suggestion fits, 4 where neither fits), 10 ambiguous, 29 false flags. Owner 8 of 11 real errors (43 to 90%), judge 53 of 89 (49 to 69%). Owner and judge agree on the scored category for 8 of the 11 items they both judged | About 3 in 10 flagged rows were correctly labelled, so "cleaned" also drops some correct, hard rows. That is a second reason, beside the cleaner's embedding bias, to read the flagged-removed lift (+1.7 to 2.2 points) as an upper bound on what label noise costs. 8 of 11 is too few items to say how reliable the judge is |
| 23 Nov 2026 | Jeff intents: 14, in `configs/jeff.yaml`. 12 map to one Jeff tool each; `safe_to_spend` and `upcoming_bills` are answered from Jeff's state block; `month_spending` covers this month (state block) and past months (`get_month`) | Section 6.1 asks for 8 to 15 intents that map to the real tools. Read from `JEFF_TOOL_SPECS` in the Velm backend (read only; nothing in Velm changed). The descriptions are the only definition of each intent, shared by the writer, the checker and S4's Jeff prompt |
| 23 Nov 2026 | Jeff set written by a separate Claude agent at the owner's request, not by the owner: 320 queries (20 per intent, 40 OOS: 15 unrelated, 25 finance-adjacent), split at random (seed 0) into 140 support and 180 test. No support-test pair within an intent is a near-duplicate (highest difflib ratio 0.68). A second agent labelled all 320 blind and agreed on 320 of 320, marking 6 unsure (`results/jeff_label_check.md`) | Owner's request. The writer saw only the intent definitions and a style brief, never the project's code, data, models or results. Wording that called the set owner-written was changed (sections 0, 2.5 R7, 3, 4 RQ6, 6, 10, 12.1); the hypotheses' substance is unchanged. Limitation: model-written and probably cleaner than real traffic, and S4 is a Claude model too |
| 23 Nov 2026 | Thresholds drafted by Claude at the owner's request: N = 60, M = 1.0, K = 5, F = 10, D = 20, L = 100 ms (p95). E2 on test: training effect only. S4 at M5: few-shot on Banking77 test plus both variants on the Jeff test split, proposed cap US$10 | The owner asked Claude to choose. Reasons per value in `results/preregistration_draft.md`. Not in force until the owner confirms it and it is saved as `results/preregistration.md`, so every test-split path still refuses |

---

## 19. Start here

1. Read this file end to end, section 0 first.
2. Ask the owner for anything unchecked in section 15 that blocks the current milestone.
3. Begin M0. Do not skip ahead to modelling before the splits are fixed and saved.
4. Get to the minimum submittable version (end of M1) before anything else. From then on, the submission notebook runs end to end at the close of every session.
