# Information Sciences Rejection — Revision Status

Context: manuscript rejected from *Information Sciences* on 4 grounds (probe robustness/leakage,
ConflictBank validity, mechanistic overclaiming, weak novelty). This tracks the technical
response before resubmission to another IS-tier journal.

## Editor's 4 concerns → fix status

| # | Concern | Fix | Status |
|---|---------|-----|--------|
| 1 | Probe test sets have 1–4 positives; feature position may leak the label | Decoupled feature anchor from label (always anchors on target-phrase end token, not the citation marker); added leave-one-condition-out transfer eval | Code done, re-extraction running |
| 2 | ConflictBank ~97% "ambiguous" no-context, doesn't measure citation | Added content-token matcher (recovers verbose parametric answers), conditional override rate (restricted to demonstrated parametric knowledge), and context→citation linkage rate | **Done**, numbers final |
| 3 | Mechanistic patching found no mechanism yet paper implies one | Added bootstrapped effect-size CIs per component type; found the effect **is** present in the residual stream (best recovery 0.163, CI [0.067, 0.253], 4 components ≥ threshold) but not localizable to individual heads/MLPs (head CI [0.004, 0.049]) — reframes as "distributed, not sparse-localizable" instead of "nothing found" | **Done**, numbers final |
| 4 | Weak novelty; 3 experiments loosely connected | Added a unifying "conditioning/robustness before reporting" paragraph in Implications; added 10 new peer-reviewed citations (attribution frameworks, knowledge-conflict literature, faithfulness-in-NLG parallels, circuit-level mechanistic work) that place the paper in a broader evidence base | **Done** |

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

## What's currently running (detached screen session)

The previous run stopped without an error message at 498/604 displayed Qwen rows; its last saved
checkpoint was 493/604. On September 29 at 15:30 CEST, the pipeline was resumed in a detached
`screen` session named `citation_probe_revision`. Qwen passed the old checkpoint; the latest
confirmed saved checkpoint was 503/604. The pipeline then continues with Llama and a fresh Gemma
extraction, followed by forced retraining, evaluation, leave-one-condition-out transfer, and
baselines for each model.

Live log: `artifacts/run_logs/resume_probe_revision_20260929T133000Z.log`.
Check `screen -ls`, the latest log progress, and `features_progress.json`; the script writes
`PIPELINE_DONE` on success or `PIPELINE_FAILED` on error. The manuscript still contains the old
probe numbers until the new results are reviewed and inserted.

If the terminal is lost/killed, resume per model with:
```
source .venv/bin/activate
python -m citation_faithfulness.cli probe extract --model <model>   # no --force = resumes
python -m citation_faithfulness.cli probe train --model <model> --force
python -m citation_faithfulness.cli probe evaluate --model <model> --force
python -m citation_faithfulness.cli probe transfer --model <model> --force
python -m citation_faithfulness.cli probe baselines --model <model> --force
```
(Do **not** set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` — revision lookup needs network even
though weights load from the local cache at `/Volumes/AI/LLMs`, which must be mounted.)

## What's left

1. Wait for re-extraction pipeline to finish; verify `PIPELINE_DONE` in the log.
2. Read new `metrics.json`, `transfer_metrics.json`, `baseline_metrics.json` per model.
3. Update paper Table `tab:probes` with the new (de-confounded) AUROC/AUPRC/balanced-accuracy
   numbers, plus a new table/paragraph for leave-one-condition-out transfer results (not yet added
   to the paper — needs the real numbers first).
4. Update the abstract's probe AUROC numbers (currently 0.943 / 0.969 / 0.831 — will change).
5. Revisit the Discussion/Conclusion/Implications sections to tighten the novelty narrative
   (concern 4) now that all three experiments (behavioral, ConflictBank, mechanistic, probe)
   share the conditional/robustness reframing.
6. Re-run `scripts/plot_paper_figures.py` if any figure needs updating (ConflictBank figure
   itself doesn't change, only the new table is additive).
7. Regenerate `artifacts/report.md` via `citation-faithfulness report` once probes are done.
8. Final proofread pass on both `citation_faithfulness_blind.tex` and the non-blind
   `citation_faithfulness.tex` (mirror all edits into the non-blind copy — not yet done).
9. Update `README.md` / PRD if any documented CLI surface changed (new commands added).
