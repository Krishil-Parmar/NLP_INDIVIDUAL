---
papersize: a4
geometry: margin=2cm
fontsize: 10pt
header-includes:
  - |
    ```{=latex}
    \usepackage{caption}
    \captionsetup{font=small,skip=3pt}
    \makeatletter
    \renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-1.6ex \@plus -.4ex \@minus -.2ex}{.4ex}{\normalfont\normalsize\bfseries}}
    \makeatother
    \setlength{\LTpre}{6pt}
    \setlength{\LTpost}{2pt}
    \usepackage{etoolbox}
    \AtBeginEnvironment{longtable}{\small}
    ```
---

```{=latex}
\begin{center}
{\large\bfseries Router Before the LLM}\\[2pt]
A cost-aware intent router for a personal finance assistant\\[2pt]
Krishil Parmar (BS24DMU022) \quad November 2026 \quad \url{https://github.com/Krishil-Parmar/NLP_INDIVIDUAL}
\end{center}
\vspace{-4pt}
```

<!--
Prose drafted by Claude Code at the owner's request on 24 Nov 2026 (disclosed in section 6 and report/ai_use_log.md).
Word budgets per section (from the M1 skeleton): 1: 130, 2: 120, 3: 130, 4: 110, 5: 150, 6: 90.
Build:  uv run python scripts/build_summary.py          (the normal build)
        uv run python scripts/build_summary.py --fill   (only for placeholders written as [TODO N words: ...])
Hard limit: 2 pages (the build fails above that).
Layout follows README section 12.1; the rubric category for each part is in brackets.
-->

## 1. Problem and constraints

Jeff is the assistant in a personal finance app. Every query it sends to its LLM (Claude Sonnet 4.6) spends a hard per-user budget and time against a 900 ms time-to-first-token target, yet many queries are routine requests for one of a few tools. This project puts a small local model, the router, in front of the LLM: it handles a query itself when confident and escalates it otherwise. **Research question:** how many LLM calls can a small router safely remove, and at what error cost? **Headline metric:** coverage at matched risk, the largest share of queries the router answers itself while its error rate on them stays at or below the LLM's error rate on the same queries. Requirements R1 to R7 (table below) turn the user's needs into checks; hypotheses and thresholds were fixed before any test run.

## 2. Data and preprocessing

**Banking77** (13,083 customer queries, 77 fine-grained banking intents) is the closest public match to Jeff's tool belt. It has no out-of-scope queries, so 1,350 from **CLINC150** are mixed in. Splits: 8,992 train, 1,000 validation (stratified, seed 42), and a test mix of 3,079 in-scope plus 1,000 out-of-scope queries (24.5%). The pipeline (`src/router/preprocess.py`, unit-tested) normalises text, removes 5 duplicates and 7 train texts that also appear in test, and sets token limits (under 0.7% truncated). CLINC150 is all lowercase, so "starts lowercase" alone separates it (AUROC 0.971); every evaluated query is therefore lowercased and stripped of punctuation. A 320-query set for Jeff's own 14 intents tests domain shift.

## 3. Systems

All five give an intent and a confidence through one interface. **S0** TF-IDF plus logistic regression: the cheapest baseline, showing what keywords alone solve. **S1** DistilRoBERTa fine-tuned with a softmax head: the standard approach. **S2** all-MiniLM-L6-v2 embeddings, answer = the label of the most similar train example (cosine): no training, and a new intent needs only examples. **S3** a paraphrase cross-encoder scoring the query against examples of S2's top 10 intents: the pairwise method, which shows the example behind each decision. **S4** Claude Sonnet 4.6 with the intent names and 2 examples per intent, temperature 0: the system being replaced, which sets the error to match and the cost to beat. **Policy:** handle when confidence reaches a threshold chosen on validation, otherwise escalate.

## 4. Results

At S4's error rate (10.27%), S2, the router chosen on validation, handles **77.47%** of test queries (95% CI 76.44 to 78.43; RQ1 met), cutting LLM cost from US\$1.58 to US\$0.35 per 1,000 queries; S2 plus the LLM is wrong on 8.65% against 10.27%. The table shows why: S4 is less accurate in scope than every router. At 1% risk S1 is better. Against a better-prompted LLM (10 similar labelled examples per query; post-hoc), S2 still covers 67.17%, but the pair makes 0.49 points more errors (5.52% to 6.01%). S3 beats S1 at spotting out-of-scope queries (RQ2 met) but not S2. Retraining without the 1.8% of rows flagged as mislabelled keeps the ranking (RQ4 met). On Jeff's intents S2 covers 1.67% (RQ6 not met).

<!-- RESULTS:START -->
| System | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage at matched risk |
| ------------------------ | ------------ | ------------ | ------------ | ------------ | -------------- |
| S0 TF-IDF + LR |  |  |  |  |  |
| S1 Fine-tuned encoder |  |  |  |  |  |
| S2 Embedding k-NN |  |  |  |  |  |
| S3 Pairwise matcher |  |  |  |  |  |
| S4 LLM few-shot |  |  |  |  |  |

: Banking77 test, in-scope queries, percent, mean ± standard deviation over five seeds. Filled in by `scripts/build_summary.py` from `results/`.
<!-- RESULTS:END -->

![Risk-coverage curves on Banking77 test (format-controlled, seed 0). The dotted line is few-shot S4's error rate, the matched risk; dots mark policy v1's frozen thresholds.](HEADLINE_FIGURE){width=62%}

<!-- REQUIREMENTS:START -->
<!-- REQUIREMENTS:END -->

## 5. What it does, why it works, limitations

**What it does.** A query typed into Jeff's chat screen is routed in about 6 ms on a laptop CPU: answered directly, with the matched example shown, or escalated to the LLM, with a running count of calls and dollars saved. **Why it works.** Banking77 is dense with paraphrases: 97.7% of in-scope test queries have a train example at cosine 0.7 or more, so the similarity to that example is a usable confidence. **Limitations.** (1) Out-of-scope queries come from another dataset, so they may be easier to spot than real ones. (2) At the frozen threshold the router answers 111 of 1,000 out-of-scope queries, the LLM 24; a stricter threshold (0.85) fixes this at 59% coverage. (3) The Jeff set is small, Claude-written and graded against a Claude LLM; only 11.4% of its in-scope queries have a close example, and fine-tuning on its 140 examples recovers part of the gap. (4) 61 of 100 audited label flags were real errors. (5) One domain, English only, no slot filling.

## 6. Code, how to run, AI use, references

**Code** (repository linked at the top): `src/router/` (systems, metrics, policy), one script per experiment in `scripts/`, 115 unit tests, pinned dependencies. **How to run:** open `notebooks/00_submission.ipynb` and choose Restart and Run All. The default `FAST = True` needs no GPU, API key or paid call, runs in about 2 minutes on a laptop, and shows every reported number from `results/`. **AI use:** Claude Code (Anthropic) wrote the code, ran the experiments and drafted this summary at my direction; Claude agents also wrote the Jeff set and judged 89 of the 100 audit items. Every session is logged in `report/ai_use_log.md`.

**References.** Casanueva et al. (2020), Banking77. Larson et al. (2019), CLINC150, EMNLP. Zhang et al. (2020), discriminative nearest neighbour few-shot intent detection, EMNLP. Hendrycks and Gimpel (2017), maximum softmax probability baseline. Geifman and El-Yaniv (2017), selective classification. Guo et al. (2017), temperature scaling. Northcutt et al. (2021), confident learning.
