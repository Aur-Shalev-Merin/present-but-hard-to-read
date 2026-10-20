# Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs

Code, pre-registrations and results for the paper by Aur Shalev Merin (arXiv link to be added once announced).

The paper asks what a probe at the start of a language model's answer can and cannot tell you. A digit copied from
the question is clearly present there (minimal pairs read it at about 0.9) but a probe that sees one prompt at a time
barely reads it, even with thousands of training questions; and every varied token leaves a trace, so a successful
probe is ambiguous too. It gives a protocol for testing computed values against copies and mixtures of the inputs,
with a synthetic null for generic mixing, and applies it to the looped model Ouro-1.4B: a small, sign-following
signal for an unwritten intermediate that Ouro does not use to write its answer.

## What is here

| folder | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex` is the master text), bibliography (`main.bib`, and `main.bbl` as submitted to arXiv), style files, figures and the built PDF |
| `preregs/` | every pre-registration: design, predictions, reading rules, frozen file hashes, amendments, and the RESULT section added after each run (as in the development repository, except that a few notes saying where a review point came from were reworded) |
| `results_docs/` | plans and result write-ups of the accuracy checks and the exploratory analyses |
| `code/` | experiment scripts, the Kaggle kernel builders (`build_kaggle_vNNN.py`) and the item files, byte-identical (LF) to the frozen versions; `code/frozen_versions/` keeps an earlier frozen version of `loop_grad_probe.py` that v178 and v186 ran with (later studies used the current file, which only adds Huginn and gradient-probe options) |
| `results/` | the analysis outputs and logs each table and figure is taken from (`results_vNNN/`) |
| `tools/verify_hashes.py` | checks the sha256 prefixes listed in each pre-registration against `code/` |
| `FREEZES.md` | for each pre-registration: freeze commit, commit time, when it first reached GitHub, and the first treatment run |

## Studies

Internal IDs (vNNN) are the experiment numbers used in the scripts, pre-registrations and result folders.

| study | question (one line) | prereg | result doc | scripts | results | paper section |
|---|---|---|---|---|---|---|
| v176 | Does Ouro's answer-only GSM8K accuracy rise with loops? (Huginn as a check) | `results_docs/v176_plan.md` | same | `v176_*` | `results_ouro_rescore/` | App. A |
| v177 | Does the gain hold on GSM-Symbolic variants? | `results_docs/v177_plan.md` | same | `v177_analyze.py` (run in a Colab notebook not included) | in the plan | App. A |
| v178 | Can a probe read an intermediate value at the last prompt token? | `preregs/prereg_v178_*` | in the prereg | `v178_*` | `results_v178/` | 3.1 |
| v179, v180 | Where is a copied digit readable, by position; bare arithmetic (exploratory) | none | `results_docs/v179_*`, `v180_*` | `v179_*`, `v180_*` | `results_v179/`, `results_v180/` | 3.1 |
| v181, v184, v185 | Minimal pairs; irrelevant number; colour word; swamping measurements (exploratory) | none | `results_docs/v181_*`, `v184_*`, `v185_*` | `v181_*`, `v184_*`, `v185_*` | `results_v181/`, `results_v184/`, `results_v185/` | 3.1, 3.2, App. B |
| v186 | Study 1: is the unwritten a+b readable beyond copies? | `preregs/prereg_v186_*` | in the prereg | `v186_*` | `results_v186*/` | 5.1, 5.2 |
| v188 | Study 2: confirmation on new templates, with answer-line controls | `preregs/prereg_v188_*` | in the prereg | `v188_*` | `results_v188*/` | 5.1, 5.2 |
| v194 | Study 3: does the better-read sign follow the operation (Part A); is a+b used (Part B, patching)? | `preregs/prereg_v194_*` | in the prereg | `v194_*` | `results_v194*/` | 5.3 |
| v199 | Is a running total readable at the end of the sentence giving b? | `preregs/prereg_v199_*` | in the prereg | `v199_*` | `results_v199/` | 5.3 |
| v202 | Do minimal pairs and swamping replicate in Qwen2.5-1.5B? | `preregs/prereg_v202_*` | in the prereg | `v202_*` | `results_v202/` | 3.2 |
| v205 | Robustness: overlap nulls, MLP probe, random perturbations, Qwen many-question curve, Study 3 with carry/borrow | `preregs/prereg_v205_*` | in the prereg | `v205_*` | `results_v205/` | 3.2, 3.3, 5.3 |
| v206 | Ouro many-question curve | `preregs/prereg_v206_*` | in the prereg | `v206_*` | `results_v206/` | 3.2 |
| v207 | Input-embedding distances of digits, words and noise (descriptive) | none | `results_docs/v207_input_distance.md` | `v207_input_distance.py` | `results_v207/` | 3.3 |
| v208 | Linear response of the last token to the varied token | `preregs/prereg_v208_*` | `results_docs/v208_result.md` | `v208_*` | `results_v208/` | 3.3, App. B |
| v209 | Patching a+b before the answer forms (loop 1, clamp, answer ablated) | `preregs/prereg_v209_*` | in the prereg | `v209_*` | `results_v209/` | 5.4 |
| v212 | Calibrated synthetic null for the protocol's mixture tests | `preregs/prereg_v212_*` | in the prereg | `v212_mixture_null.py` | `results_v212/` | 5.4 |

## Reproducing

- **Model runs** went to Kaggle (T4 GPUs, or CPU for analysis). Each `build_kaggle_vNNN.py` writes a kernel folder
  (`run.py` with the scripts embedded, plus `kernel-metadata.json`) that `kaggle kernels push` runs. The kernel ids in
  the builders name the account the runs used; change them to your own. Kernels install `transformers==4.54.1` and
  `datasets`; Ouro-1.4B and Qwen2.5-1.5B are downloaded from Hugging Face.
- **Analyses** are plain Python on the downloaded outputs (`python code/vNNN_analyze.py <dir>`), and each prereg says
  which command produced its RESULT section. The probes use scikit-learn.
- **Figures:** run `make_ouro_note_figs.py` and `make_ouro_note_fig_curve.py` from a folder that contains the
  `results_vNNN/` directories (for example `cd results && python ../code/make_ouro_note_fig_curve.py`).
- **Paper:** `cd paper && tectonic -X compile main.tex` (or pdflatex with `main.bbl`).
- `environment.yml` lists the packages.

Two exploratory scripts (`v178_diag_exploratory.py`, `v207_input_distance.py`) had an absolute local path replaced by
a relative one; neither is listed with a hash in any pre-registration. Every other file is unchanged.

## Pre-registration

Each frozen study's design, predictions and reading rules were committed before its first treatment run, and the
paper reports every outcome, including the misses (Appendix C lists the status of every analysis). `FREEZES.md` gives
the freeze commits and, from GitHub's own records, when each reached GitHub relative to the first run. The
development repository is private; commits are available to reviewers on request.

## License

Code: MIT (`LICENSE`). Paper text, pre-registrations and result documents: CC BY 4.0.

## Citation

See `CITATION.cff`.
