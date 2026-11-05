# Copied, Mixed or Computed? Testing Latent Intermediate Values with Minimal Pairs

Code, pre-registrations and results for the paper by Aur Shalev Merin: [arXiv:2610.28498](https://arxiv.org/abs/2610.28498).
Version 1 was posted under the title "Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs"; the
current version keeps the same arXiv ID.

Probing studies of latent reasoning read a model's state where the answer begins and take a successful probe as
evidence that an intermediate value was computed, and a failed one as evidence that it was not. Both readings
mislead. Using prompts that differ in a single token (minimal pairs), a digit copied from the question is present at
that position in two looped and two standard language models, while a probe that sees one prompt at a time reads it
at no more than 0.52 even after 45,000 training questions; and every token we varied leaves such a trace, so a probe
for a computed value may be reading the inputs. The paper gives a protocol that tests a candidate value against
copies of the inputs (a copy ceiling), against generic mixing of them (a share-based operation swap whose bar is set
on synthetic mixtures), and for use (a clamp with the copy projected out), and checks each test where the answer is
known: values a model is about to write or reads a few tokens later, and an unwritten value used eight tokens later
in small transformers trained from scratch. In the looped Ouro-1.4B the protocol finds a small signal for the
unwritten intermediate whose sign follows the operation, but it cannot be told apart from the answer with c added
back, no running total shows before c is read, and whether the value is used is unresolved. For a published claim
that language models compute sub-answers only while writing a chain of thought (Kudo et al.), the protocol cannot
tell.

## What is here

| folder | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex` is the master text, the arXiv copy), bibliography (`main.bib`, and `main.bbl` generated from it), the style files it uses, figures and the built PDF |
| `preregs/` | every pre-registration: design, predictions, reading rules, frozen file hashes, amendments, and the RESULT section added after each run (as in the development repository, except that a few notes saying where a review point or a literature check came from, or how a freeze was approved, were reworded, and mentions of a venue removed) |
| `results_docs/` | plans, designs and result write-ups of the accuracy checks, the exploratory analyses and two studies whose result sits outside the prereg (v220, and the designs of v230 and v231) |
| `code/` | experiment scripts, the Kaggle kernel builders (`build_kaggle_vNNN.py`) and the item files, byte-identical (LF) to the frozen versions; `code/frozen_versions/` keeps earlier frozen versions where a file changed after its freeze: `loop_grad_probe.py` as v178 and v186 ran it (later studies used the current file, which only adds Huginn and gradient-probe options), and the builders of v213 and v216 as frozen (the current builders add the kernel modes used later for the split many-question run and the post hoc probes) |
| `results/` | the analysis outputs and logs each table and figure is taken from (`results_vNNN/`); model checkpoints, saved states and large plan archives are left out |
| `tools/verify_hashes.py` | checks the sha256 prefixes listed in each pre-registration against `code/` |
| `FREEZES.md` | for each pre-registration: freeze commit, commit time, when it first reached GitHub, and the first treatment run |

## Studies

Internal IDs (vNNN) are the experiment numbers used in the scripts, pre-registrations and result folders. Paper
sections refer to the current version (Appendix F lists the status of every analysis).

| study | question (one line) | prereg | result doc | scripts | results | paper section |
|---|---|---|---|---|---|---|
| v176 | Does Ouro's answer-only GSM8K accuracy rise with loops? (Huginn as a check) | `results_docs/v176_plan.md` | same | `v176_*` | `results_ouro_rescore/` | App. A |
| v177 | Does the gain hold on GSM-Symbolic variants? | `results_docs/v177_plan.md` | same | `v177_analyze.py` (run in a Colab notebook not included) | in the plan | App. A |
| v178 | Can a probe read an intermediate value at the last prompt token? | `preregs/prereg_v178_*` | in the prereg | `v178_*` | `results_v178/` | 3, App. B.1 |
| v179, v180 | Where is a copied digit readable, by position; bare arithmetic (exploratory) | none | `results_docs/v179_*`, `v180_*` | `v179_*`, `v180_*` | `results_v179/`, `results_v180/` | 3, App. B.1 |
| v181, v184, v185 | Minimal pairs; irrelevant number; colour word; swamping measurements (exploratory) | none | `results_docs/v181_*`, `v184_*`, `v185_*` | `v181_*`, `v184_*`, `v185_*` | `results_v181/`, `results_v184/`, `results_v185/` | 3, App. B.1-B.3 |
| v186 | Study 1: is the unwritten a+b readable beyond copies? | `preregs/prereg_v186_*` | in the prereg | `v186_*` | `results_v186*/` | 6, App. C.1, C.2 |
| v188 | Study 2: confirmation on new templates, with answer-line controls | `preregs/prereg_v188_*` | in the prereg | `v188_*` | `results_v188*/` | 6, App. C.1, C.2 |
| v194 | Study 3: does the better-read sign follow the operation (Part A); is a+b used (Part B, patching)? | `preregs/prereg_v194_*` | in the prereg | `v194_*` | `results_v194*/` | 6, App. C.3 |
| v199 | Is a running total readable at the end of the sentence giving b? | `preregs/prereg_v199_*` | in the prereg | `v199_*` | `results_v199/` | App. C.3 |
| v202 | Do minimal pairs and swamping replicate in Qwen2.5-1.5B? | `preregs/prereg_v202_*` | in the prereg | `v202_*` | `results_v202/` | 3, App. B.2 |
| v205 | Robustness: overlap nulls, MLP probe, random perturbations, Qwen many-question curve, Study 3 with carry/borrow | `preregs/prereg_v205_*` | in the prereg | `v205_*` | `results_v205/` | 3, App. B.2, B.3, C.3 |
| v206 | Ouro many-question curve | `preregs/prereg_v206_*` | in the prereg | `v206_*` | `results_v206/` | 3, App. B.2 |
| v207 | Input-embedding distances of digits, words and noise (descriptive) | none | `results_docs/v207_input_distance.md` | `v207_input_distance.py` | `results_v207/` | App. B.3 |
| v208 | Linear response of the last token to the varied token | `preregs/prereg_v208_*` | `results_docs/v208_result.md` | `v208_*` | `results_v208/` | App. B.3, B.4 |
| v209 | Patching a+b before the answer forms (loop 1, clamp, answer ablated) | `preregs/prereg_v209_*` | in the prereg | `v209_*` | `results_v209/` | 6, App. C.4 |
| v212 | Calibrated synthetic null for the protocol's mixture tests | `preregs/prereg_v212_*` | in the prereg | `v212_mixture_null.py` | `results_v212/` | App. C.4 |
| v213 | Scale-up: minimal pairs and the perturbation null in Qwen2.5-7B and Ouro-2.6B; many-question curve to 45,249 questions with Orca-Math | `preregs/prereg_v213_*` | in the prereg | `v213_*` | `results_v213/` | 3, App. B.2, B.3 |
| v214 | Does one reference variant do the work of the ten-variant minimal-pair contrast? (four models) | `preregs/prereg_v214_*` | in the prereg | `v214_one_ref.py` | `results_v214/` | App. B.1 |
| v215 | Re-test of Kudo et al. with a chain of thought: minimal pairs, copy ceiling, operation swap, mixing null, clamp, their probe | `preregs/prereg_v215_*` | in the prereg | `v215_*` | `results_v215/` | 7, App. D |
| v216 | The same without a chain of thought (answer only), with the answer as a confound; their probe in this format | `preregs/prereg_v216_*` | in the prereg | `v216_*` | `results_v216/` | 7, App. D |
| v220 | Training order in small transformers: does a op b appear before the answer? | `preregs/prereg_v220_*` | `results_docs/v220_result.md` (design: `v220_design.md`) | `v220_*` | `results_v220/` | App. C.4, C.6 |
| v221 | Positive control: the protocol where Qwen2.5-7B writes b in the chain; their probe on the copied operands | `preregs/prereg_v221_*` | in the prereg | `v221_*` | `results_v221/` | 5, App. D |
| v222 | Positive control at a distance: clamp at b's written digit, outcome a four tokens later | `preregs/prereg_v222_*` | in the prereg | `v222_*` | `results_v222/` | 5 |
| v223 | Why the accuracy-based swap failed: pair-code ceiling, random pair functions, variance shares (exploratory) | none | in v224's prereg | `v223_swapdiag.py` | `results_v223/` | 5 |
| v224 | Share-based operation swap: frozen statistic, bar from 420 null worlds, fresh controls | `preregs/prereg_v224_*` | in the prereg | `v224_*` | `results_v224/` | 4, 5 |
| v227 | Replacement control for the share swap (Qwen2.5-3B dropped at its accuracy pilot; Mistral-7B, the b step) | `preregs/prereg_v227_*` | in the prereg | `v227_run.py` | `results_v227/` | 5 |
| v228 | Re-score of the targets with the share swap (Kudo et al.'s format, answer only, Ouro Study 3) | `preregs/prereg_v228_*` | in the prereg | `v228_rescore.py` | `results_v228/` | 6, 7 |
| v229 | Ouro P1 with the carry and borrow regressed; null on Study 3's design | `preregs/prereg_v229_*` | in the prereg | `v229_robust.py` | `results_v229/` | 6 |
| v230 | An unwritten, used value in small transformers trained from scratch (ground truth from DAS): clamp and share swap; post hoc class-mean clamp and readability check | `preregs/prereg_v230_*` | in the prereg (design: `results_docs/v230_design.md`) | `v230_*` | `results_v230/` | 5, App. E |
| v231 | Clamp power as the value gets as faint as Ouro's | `preregs/prereg_v231_*` | in the prereg (design: `results_docs/v231_design.md`) | `v231_power.py` | `results_v231/` | 6, App. E |
| v232 | Ouro: a running total at the end of b's sentence, before c is read (share swap with a design-matched null; clamp with the copies projected out) | `preregs/prereg_v232_*` | in the prereg | `v232_sb.py` (with `v199_patch.py`) | `results_v232/` | 6, App. C.3 |
| post hoc | Ouro's patches and clamps (v194, v209) against the random subspace alone | none (post hoc, after v231) | output only | `posthoc_ab_vs_rand.py` | `posthoc_ab_vs_rand.txt` | 6, App. C.3 |

## Reproducing

- **Model runs** went to Kaggle (T4 GPUs, or CPU for analysis). Each `build_kaggle_vNNN.py` writes a kernel folder
  (`run.py` with the scripts embedded, plus `kernel-metadata.json`) that `kaggle kernels push` runs. The kernel ids in
  the builders name the account the runs used; change them to your own. Kernels install `transformers==4.54.1` and
  `datasets`; Ouro-1.4B, Ouro-2.6B, Qwen2.5-1.5B, Qwen2.5-7B and Mistral-7B-v0.1 are downloaded from Hugging Face.
  The small transformers of v220, v230 and v231 are trained inside their kernels; their checkpoints are not included
  (v230 checks each retrained model against the checkpoint hashes in its prereg).
- **Analyses** are plain Python on the downloaded outputs (`python code/vNNN_analyze.py <dir>`), and each prereg says
  which command produced its RESULT section. The probes use scikit-learn.
- **Post hoc check of Ouro's patches against random:** `python code/posthoc_ab_vs_rand.py` reads `results/results_v194`
  and `results/results_v209` and prints `results/posthoc_ab_vs_rand.txt`. It was written after the v231 power study
  and is not pre-registered.
- **Figures:** run `make_ouro_note_figs.py` and `make_ouro_note_fig_curve.py` from a folder that contains the
  `results_vNNN/` directories (for example `cd results && python ../code/make_ouro_note_fig_curve.py`).
- **Paper:** `cd paper && tectonic -X compile main.tex` (or pdflatex and bibtex).
- `environment.yml` lists the packages.

Four scripts had an absolute local path replaced by a relative one: the exploratory `v178_diag_exploratory.py` and
`v207_input_distance.py`, and the post hoc `v230_posthoc_classmean.py` and `posthoc_ab_vs_rand.py` (both read their
inputs from next to the script or from `results/`). None is listed with a hash in any pre-registration. Every other
file is unchanged.

## Pre-registration

Each frozen study's design, predictions and reading rules were committed before its first treatment run, and the
paper reports every outcome, including the misses (Appendix F lists the status of every analysis). `FREEZES.md` gives
the freeze commits and, from GitHub's own records, when each reached GitHub relative to the first run. The
development repository is private; commits are available to reviewers on request.

## License

Code: MIT (`LICENSE`). Paper text, pre-registrations and result documents: CC BY 4.0.

## Citation

See `CITATION.cff`.
