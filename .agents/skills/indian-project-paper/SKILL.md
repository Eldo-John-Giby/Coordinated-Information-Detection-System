---
name: indian-project-paper
description: Turns an implemented undergraduate technical project into a credible, evidence-backed academic research paper tailored for Indian university, UGC-CARE, Scopus, or IEEE conference/journal submissions. Activates on `/indian-project-paper` and related research-writing queries.
license: MIT
version: 1.0.0
---

# Indian Undergraduate Project Research Paper

## Primary objective
Turn an implemented undergraduate technical project into a credible, evidence-backed academic manuscript. Never treat a project report as a research paper without identifying a genuine research question and defensible contribution.

## Phase 1: Understand the project
Before writing:
1. Inspect the repository, README, source code, notebooks, configuration files, datasets, experiment scripts, and existing reports.
2. Identify the problem statement, research question, baseline methods, proposed method, technical contribution, and limitations.
3. Extract actual experimental results from logs, CSVs, notebooks, and reproducible runs.
4. Create PROJECT_FACTS.md containing verified implementation facts, result provenance, assumptions, and unknowns.
5. Ask the user for missing essential information instead of inventing it.
Do not modify the project's core implementation merely to make the paper appear stronger.

## Phase 2: Literature review
1. Search for relevant peer-reviewed papers and credible preprints.
2. Prioritize original research, recent relevant work, and seminal papers.
3. Verify titles, authors, publication details, DOI, and source URLs.
4. Compare methods, datasets, evaluation metrics, limitations, and findings.
5. Build a literature matrix and identify a specific, evidence-supported research gap.
6. Never claim that a method is the first of its kind without adequate evidence.

## Phase 3: Research design
Document:
- Research objectives and hypotheses, where appropriate.
- Dataset origin, licensing, preprocessing, and class distribution.
- Model architecture and implementation details.
- Baselines and fair comparison conditions.
- Training, validation, and test splits.
- Hyperparameters, random seeds, hardware, and software versions.
- Evaluation metrics appropriate to the task.
- Ablation studies and statistical tests where justified.
- Reproducibility requirements, ethical concerns, and limitations.
Distinguish proposed experiments from completed experiments. Do not report hypothetical results as measured findings.

## Phase 4: Manuscript
Use the target venue's actual template and author guidelines.

Default empirical-paper structure:
1. Title and author information
2. Abstract and keywords
3. Introduction and research contributions
4. Related work
5. Problem formulation and methodology
6. Experimental setup
7. Results and comparative analysis
8. Discussion and error analysis
9. Limitations and threats to validity
10. Conclusion and future work
11. References
12. Supplementary material or reproducibility statement, if required
Adapt the structure to the research domain and venue. Do not force every paper into the same template.

## Phase 5: Indian academic requirements
1. Ask which university, department, conference, journal, or submission scheme applies.
2. Follow its official formatting, authorship, declaration, page-limit, and submission requirements.
3. Use IEEE style only if the venue requires it.
4. Verify any claims about UGC-CARE, Scopus, Web of Science, or IEEE affiliation using authoritative current sources.
5. Never imply that publication guarantees academic credit or institutional approval.
6. Never invent supervisor approval, institutional affiliations, ethics approval, or author declarations.
7. Check applicable institutional rules on AI assistance, originality, and disclosure.

## Phase 6: Figures, tables, and results
- Generate charts from actual data and preserve reproducible scripts.
- Label axes, units, datasets, sample sizes, and evaluation conditions.
- Use tables to compare baselines under consistent settings.
- Never fabricate plots, confidence intervals, p-values, or benchmark results.
- Report negative results and important failure cases.
- Explain what each result demonstrates and what it does not establish.

## Phase 7: Academic integrity and citations
- Verify every reference against a reliable scholarly record.
- Ensure every in-text citation has a matching bibliography entry and vice versa.
- Cite the original source for methods, datasets, and borrowed claims.
- Avoid plagiarism, citation padding, fabricated sources, and unsupported paraphrases.
- Do not manipulate experiments to obtain desired significance.
- Do not promise an undetectable AI-written paper or a guaranteed acceptance.

## Phase 8: Independent review
Review the draft as:
1. A domain expert evaluating novelty and technical depth.
2. A methods reviewer evaluating experimental validity.
3. A skeptical reviewer evaluating unsupported claims and reproducibility.
4. A copy editor checking clarity, structure, grammar, and formatting.
Resolve critical issues before finalizing.

## Deliverables
Create:
- `PROJECT_FACTS.md`
- `literature_matrix.csv`
- `research_gap.md`
- `manuscript.md`
- references and citation verification report
- reproducible figures, tables, and scripts
- `Known-gaps.md` listing unresolved issues
- submission-readiness checklist
Keep the manuscript editable. Generate DOCX, LaTeX, or PDF only after confirming the target venue's requirements.
