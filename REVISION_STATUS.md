# Information Sciences Rejection — Revision Status

Context: manuscript rejected from *Information Sciences* on 4 grounds (probe robustness/leakage,
ConflictBank validity, mechanistic overclaiming, weak novelty). This tracks the technical
response before resubmission to another IS-tier journal.

## Editor's 4 concerns → fix status

| # | Concern | Fix | Status |
|---|---------|-----|--------|
| 1 | Probe test sets have 1–4 positives; feature position may leak the label | Decoupled feature anchor from label; re-extracted all three models and ran leave-one-condition-out transfer. Corrected probes scored below the strongest baseline in every model; paper now reports this negative result. | Leakage concern addressed; robust screening claim unsupported |
| 2 | ConflictBank ~97% "ambiguous" no-context, doesn't measure citation | Added content-token matcher, conditional override rate, and context→citation linkage rate. No-context ambiguity remains 1,994/2,000 Llama, 1,942/2,000 Qwen, and 1,916/2,000 Gemma; demonstrated-knowledge subsets are 5, 56, and 71. | Partial; causal citation reliance unproven |
| 3 | Mechanistic patching found no mechanism yet paper implies one | Added bootstrapped effect-size CIs. The best residual recovery is 0.163 (CI [0.067, 0.253]); four residual positions have mean recovery ≥ 0.10, while no individual head or MLP write does. Scaling still corrected 0/7 missed and 0/58 spurious citations. | Overclaim reduced; mechanism unvalidated |
| 4 | Weak novelty; 3 experiments loosely connected | Added a shared conditioning/robustness argument and peer-reviewed context, but no new decisive comparison against Wallat et al. or independent causal result. | Open for another selective journal |

## What's done

- **Code**: `probes/features.py` (label-independent anchor `_locate_end`), `probes/evaluate.py`
  (`transfer()` = leave-one-condition-out), `data/conflictbank.py` (`answer_contains`,
  `_cites_doc1`, `reclassify()`, richer `metrics()`), `mechanistic/patching.py`
  (`null_analysis()` with bootstrapped CIs), new CLI commands (`conflictbank reclassify`,
  `probe transfer`, `mechanistic null-analysis`).
- **Tests**: `tests/test_revision_changes.py` (7 new tests) — all 27 project tests pass.
- **No-model recomputation** (from existing saved artifacts, no LLM rerun needed):
  - ConflictBank reclassified + new conditional metrics — **final numbers in hand**.
  - Mechanistic null analysis with bootstrap CIs — **final numbers in hand**.
- **Paper edits already applied** to `Elsevier/citation_faithfulness_blind.tex` **and mirrored into**
  `Elsevier/citation_faithfulness.tex` (verified in sync — the only remaining diffs between the two
  files are 3 pre-existing, unrelated ones: an illustrative example paragraph, a dataset href label,
  and a `\begingroup`/`\endgroup` wrapper):
  - ConflictBank methodology: added conditional CMR formula + context-citation rate definition +
    content-token matcher description.
  - ConflictBank results: added Table `tab:conflictbank-conditional` (conditional override %,
    context-citation rate per model) + explanatory paragraph.
  - Mechanistic methodology: added sentence describing the bootstrap procedure.
  - Mechanistic results: added bootstrapped-CI paragraph reframing the null as
    residual-stream-present-but-not-localized.
  - Probe methodology: rewrote decision-position description to the label-independent anchor;
    added leave-one-condition-out transfer description.
  - Discussion: rewrote the ConflictBank paragraph (conditional framing) and mechanistic paragraph
    (distributed-not-absent finding).
  - Implications: added a new unifying paragraph — the paper's response to concern 4 (novelty) —
    framing all four experiments as sharing one "conditioning/robustness before reporting" discipline.
  - Limitations: removed the now-false "did not test cross-condition probe transfer" claim; added a
    note on the small ConflictBank parametric-known subsets (Llama n=5).
  - Conclusion: reframed the mechanistic sentence to "distributed residual-stream effect" and added a
    sentence on the probe's label-independent anchor + leave-one-condition-out mitigation.
- **10 new peer-reviewed references added** to `Elsevier/references.bib` and cited in Related Work
  (both tex files, verified in sync). Each was verified against ACL Anthology / ICLR / NeurIPS /
  Computational Linguistics / TACL before adding — no fabricated citations:
  - Asai et al., *Self-RAG* (ICLR 2024) — RAG evolution paragraph.
  - Rashkin et al., *Measuring Attribution in NLG Models* / AIS framework (Computational Linguistics
    2023) and Dziri et al., *BEGIN Benchmark* (TACL 2022) — attributed-generation paragraph.
  - Maynez et al., *On Faithfulness and Factuality in Abstractive Summarization* (ACL 2020) and
    Turpin et al., *Language Models Don't Always Say What They Think* (NeurIPS 2023) — new paragraph
    paralleling citation post-rationalization with hallucination/CoT unfaithfulness elsewhere in NLG.
  - Longpre et al., *Entity-Based Knowledge Conflicts in QA* (EMNLP 2021), Neeman et al., *DisentQA*
    (ACL 2023), Xu et al., *Knowledge Conflicts for LLMs: A Survey* (EMNLP 2024), and Lin et al.,
    *TruthfulQA* (ACL 2022) — knowledge-conflicts paragraph, directly deepening the ConflictBank
    discussion.
  - Wang et al., *Interpretability in the Wild* / IOI circuit (ICLR 2023) — mechanistic paragraph,
    reinforcing the "distributed, not sparse-localized" reframing from concern 3.
- Old (confounded) probe artifacts backed up to `artifacts/probes_backup_confounded/`.
- Nine further peer-reviewed references were added and cited in both manuscript versions; both PDFs
  were rebuilt with exactly 45 references.
- Session/repo memory notes recorded in `/memories/session/citation-faithfulness-revision.md`.

## Current run status: complete (September 29, 19:39 CEST)

The previous run stopped without an error message at 498/604 displayed Qwen rows; its last saved
checkpoint was 493/604. A detached session resumed the Qwen extraction and completed all 604 rows,
then refreshed its training, evaluation, leave-one-condition-out transfer, and baselines. Its
all-model launcher was then intentionally stopped before Gemma. Llama continued in a separate
Llama-only session, completed all 479 extraction rows, and refreshed the same four downstream steps.
That session exited successfully with `LLAMA_DONE_PIPELINE_PAUSED` at 16:07 CEST. On the user's
subsequent request, Gemma was started in a detached `screen` session named `citation_probe_gemma`.
Its old features were backed up, and the new extraction completed all 1,464 rows using the
corrected feature anchor. Training, evaluation, transfer testing, and baselines finished with the
`GEMMA_PIPELINE_DONE` marker at 19:39 CEST. No probe run is active.

Logs: `artifacts/run_logs/resume_probe_revision_20260929T133000Z.log` (intentionally interrupted
all-model launcher), `artifacts/run_logs/finish_llama_then_pause_20260929T140400Z.log` (Llama
completion), and `artifacts/run_logs/finish_gemma_revision_20260929T150253Z.log` (completed Gemma
run). The probe results were reviewed and inserted in both article versions;
the abstract, tables, discussion, conclusion, highlights, cover letter, and editor response now
state that the corrected probes did not exceed the strongest baseline by AUROC.

Corrected test AUROC: Llama 0.243, Qwen 0.531, Gemma 0.328. Each test set has only 1, 2, and 4
positive examples, respectively. Several leave-one-condition-out held-out sets have no positives.
`artifacts/report.md` has been regenerated from the completed metrics.

## What's left

1. Manually audit a sample of behavioral labels and repeat the probe evaluation with more positive
   cases if a reliable hidden-state monitor remains a research goal.
2. Update `README.md` / PRD if the new CLI commands should be documented for external users.

## Repurposed manuscript positioning (29 September 2026)

The article is now framed as **“Stress-Testing Citation Faithfulness in Open-Weight
Retrieval-Augmented Generation.”** Its primary result is an operational, cross-model
comparison of citation assignment after answer-bearing document insertions. The main
table now reports event, available-intervention, and recovered-statement counts separately;
the percentages use recovered statements as their denominator. ConflictBank, PopQA patching,
and hidden-state probes are explicitly auxiliary diagnostics with limited or negative
findings. The abstract, introduction, objectives, methods, results, discussion, conclusion,
highlights, and generic cover letter were updated in both identified and blind versions.

This reframing improves the match between claims and evidence, but it does **not** resolve
the novelty concern for a selective journal through a new experiment. Before claiming a
causal document-relevance effect or true citation provenance, the study still needs matched
document positions and source retention, placebo/no-insertion and source-ablation controls,
manual label auditing, and independent validation. The prior `Response_to_the_Editor` is
specific to the rejected submission and should not accompany a new journal submission.

Both article PDFs and the one-page highlights and cover-letter PDFs compiled successfully
with TeX Live. The two article PDFs are 16 pages each, cite all 45 bibliography entries,
and have no unresolved citation or reference warnings. The identified first page and the
behavioral methods page were visually checked after compilation.
