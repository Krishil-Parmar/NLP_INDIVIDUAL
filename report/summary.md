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
    ```
---

```{=latex}
\begin{center}
{\large\bfseries Router Before the LLM}\\[2pt]
A cost-aware intent router for a personal finance assistant\\[2pt]
Krishil Parmar \quad [TODO date]
\end{center}
\vspace{-4pt}
```

<!--
SKELETON (M1). Owner writes every sentence in his own words.
Each [TODO N words: ...] marks what goes there and roughly how much fits.
Build:  uv run python scripts/build_summary.py          (TODOs shown as-is)
        uv run python scripts/build_summary.py --fill   (TODOs replaced by N filler words, to test the fit)
Hard limit: 2 pages (the build fails above that).
Layout follows README section 12.1; the rubric category for each part is in brackets.
-->

## 1. Problem and constraints

[TODO 130 words: the task, the user, the per-user budget and the 900 ms time-to-first-token constraint, and the one research question. (Rubric 1)]

## 2. Data and preprocessing

[TODO 120 words: Banking77 and CLINC150 out-of-scope queries and why, split sizes, and two or three numbers from results/data_report.md. (Rubric 2)]

## 3. Systems

[TODO 130 words: one line per system (S0 to S4) saying why it is in the comparison, from README section 7.2, and the router policy in one sentence. (Rubric 2)]

## 4. Results

[TODO 110 words: coverage at matched risk and cost saved in one sentence; what the table shows; one sentence each for any stretch results. (Rubric 4)]

<!-- RESULTS:START -->
| System | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage at matched risk |
| ------------------------ | ------------ | ------------ | ------------ | ------------ | -------------- |
| S0 TF-IDF + LR |  |  |  |  |  |
| S1 Fine-tuned encoder |  |  |  |  |  |
| S2 Embedding k-NN |  |  |  |  |  |
| S3 Pairwise matcher |  |  |  |  |  |
| S4 LLM zero-shot |  |  |  |  |  |

: Banking77 test, in-scope queries, percent, mean ± standard deviation over five seeds. Filled in by `scripts/build_summary.py` from `results/`.
<!-- RESULTS:END -->

![Risk-coverage curves with LLM calls and cost saved (headline figure, added in M4).](HEADLINE_FIGURE){width=75%}

| Requirement | Check | Met? | Evidence |
| ---------------------------------------------- | ---------- | ------ | ------------------------------ |
| R1 Correct intent on in-scope queries | E1 |  |  |
| R2 OOS and uncertain queries not handled | E3, E6 |  |  |
| R3 Fewer LLM calls, error no higher | E6 |  |  |
| R4 Router latency within limit | E6 |  |  |
| R5 New intent from about 10 examples | E4 |  |  |
| R6 Stable under rewording | E5 |  |  |
| R7 Works on owner-written queries | E7 |  |  |

: Requirements from README section 2.5. (Rubric 1)

## 5. What it does, why it works, limitations

[TODO 150 words: the same three headings the demo uses. Limitations must be specific: out-of-scope data from a different source, small owner-written set, label noise, single domain. (Rubric 5)]

## 6. Code, how to run, AI use, references

[TODO 90 words: repository link; how to run notebooks/00_submission.ipynb (FAST mode, run time); AI use disclosure from report/ai_use_log.md as the unit requires. (Rubric 3)]

**References.** Casanueva et al. (2020), Banking77. Larson et al. (2019), CLINC150, EMNLP. Zhang et al. (2020), discriminative nearest neighbour few-shot intent detection, EMNLP. Hendrycks and Gimpel (2017), maximum softmax probability baseline. Geifman and El-Yaniv (2017), selective classification. Guo et al. (2017), temperature scaling. Northcutt et al. (2021), confident learning.
