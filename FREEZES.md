# Pre-registration freezes

Every study below was written up as a pre-registration (`preregs/`, or `results_docs/v176_plan.md` and
`v177_plan.md` for the two accuracy checks) and frozen in a commit before its first treatment run. The table gives,
per freeze and amendment:

- **committed:** the commit's committer time in the development repository. Git lets an author set this, so it is
  not external evidence on its own.
- **first on origin:** the earliest entry in GitHub's repository activity log (`GET /repos/{owner}/{repo}/activity`,
  a push or a branch creation) whose new head contains the commit and whose previous head does not. This is GitHub's
  own record. The log reaches back to February 2026, so it covers every study here.
- **first treatment run:** the time Kaggle recorded for the first kernel that ran treatment conditions (a pilot,
  selftest or smoke test on controls only is not a treatment run). For kernels that were re-pushed, Kaggle shows the
  latest version's time; the notes say where an earlier time was recorded during the session.

All times are UTC, ISO 8601. Commit hashes refer to the private development repository; the files they freeze are in
this repository, byte-identical (LF line endings), so the sha256 prefixes each pre-registration lists can be checked
with `tools/verify_hashes.py`.

| study | internal ID | prereg file | freeze commit | committed (UTC) | first on origin (UTC) | branch | first treatment run (UTC) | notes |
|---|---|---|---|---|---|---|---|---|
| Answer-only GSM8K accuracy by loop (Appendix A) | v176 | `results_docs/v176_plan.md` | 6685b1edea3c | 2026-10-06T02:25:53Z | 2026-10-06T17:32:19Z | feature/documentation | 2026-10-06T03:36:29Z (`astrocyte-v176-answeronly`) | Reached origin after its run. The paper claims only that this freeze was committed before the run. |
| GSM-Symbolic variants (Appendix A) | v177 | `results_docs/v177_plan.md` | dcc9473acc07 | 2026-10-06T07:40:49Z | 2026-10-06T17:32:19Z | feature/documentation | run on Colab; no external timestamp | As v176: committed before the run, pushed afterwards. |
| Intermediate value at the last prompt token (Sec. 3.1) | v178 | `preregs/prereg_v178_ouro_latent_intermediates.md` | e99fd90f3fc6 | 2026-10-06T18:13:30Z | 2026-10-10T19:51:59Z | feature/documentation | 2026-10-10T19:55:39Z (`astrocyte-v178-ouro`) | |
| Study 1 (Sec. 5.1) | v186 | `preregs/prereg_v186_ouro_unwritten_intermediate.md` | 135fa03e3e30 | 2026-10-12T18:12:07Z | 2026-10-12T18:12:57Z | feature/documentation | 2026-10-12T18:13:08Z (`astrocyte-v186-ouro`) | The pilot (05:51) ran before the freeze and is recorded in the prereg. |
| Study 2 (Sec. 5.1) | v188 | `preregs/prereg_v188_ouro_intermediate_confirmation.md` | 470f683e0606 | 2026-10-13T00:38:24Z | 2026-10-13T00:38:58Z | feature/documentation | 2026-10-13T00:39:15Z (`astrocyte-v188-ouro`) | Its bar was set after Study 1's result (stated in the paper). |
| Study 3, Part A (Sec. 5.3) | v194 | `preregs/prereg_v194_ouro_operation_swap.md` | defaaf0d19e8 | 2026-10-14T00:40:40Z | 2026-10-14T00:43:33Z | feature/documentation | 2026-10-14T00:44:20Z (`astrocyte-v194-ouro`) | |
| Study 3, Part B amendment | v194 | same | f5d8cf244884 | 2026-10-14T16:33:45Z | 2026-10-14T16:33:54Z | feature/documentation | 2026-10-14T16:34:19Z (`astrocyte-v194-cpu`, plan); 17:08:35Z (`astrocyte-v194-patch`) | Added the loop-2 patch point and equal-answer pairs after extraction, before any analysis. |
| Running total (Sec. 5.3) | v199 | `preregs/prereg_v199_ouro_running_total.md` | 071f79835027 | 2026-10-14T22:50:45Z | 2026-10-14T22:51:04Z | feature/documentation | 2026-10-14T22:51:23Z (`astrocyte-v199-ouro`) | |
| Qwen replication (Sec. 3.2) | v202 | `preregs/prereg_v202_qwen_swamping.md` | ffe382a86fad | 2026-10-15T21:34:20Z | 2026-10-15T21:34:35Z | feature/documentation | 2026-10-15T21:34:40Z (`astrocyte-v202-qwen`) | |
| Robustness, incl. Qwen many-question curve (Secs. 3.2, 3.3) | v205 | `preregs/prereg_v205_swamping_robustness.md` | 2ecfa2d0c7e6 | 2026-10-16T20:13:22Z | 2026-10-16T20:13:27Z | feature/documentation | 2026-10-16T20:13:43Z (`astrocyte-v205-b1`) | |
| v205 amendment (numerical fallback) | v205 | same | 1b01747be357 | 2026-10-16T20:20:44Z | 2026-10-16T20:20:46Z | feature/documentation | 2026-10-16T22:59:43Z (`astrocyte-v205-amain`, analysis) | Before any v205 statistic was computed. |
| Ouro many-question curve (Sec. 3.2) | v206 | `preregs/prereg_v206_ouro_template_curve.md` | 8fcab2b11318 | 2026-10-16T23:55:08Z | 2026-10-16T23:55:14Z | feature/documentation | 2026-10-16T23:55:32Z (`astrocyte-v206-b2s0`) | |
| Linear response at the varied token (Sec. 3.3, App. B) | v208 | `preregs/prereg_v208_input_sensitivity.md` | b0f3ce2a525b | 2026-10-18T20:11:44Z | 2026-10-18T20:11:47Z | feature/documentation | 2026-10-18T20:12:04Z (`astrocyte-v208-qwen0`) | |
| v208 amendment 1 | v208 | same | f581b9cc0b46 | 2026-10-19T00:31:05Z | 2026-10-19T00:31:08Z | feature/documentation | 2026-10-19T00:56:21Z (`astrocyte-v208-bqwen`) | Before any Part B statistic. |
| Patching before the answer forms (Sec. 5.4) | v209 | `preregs/prereg_v209_ouro_early_patch.md` | 821960e017f0 | 2026-10-19T16:01:12Z | 2026-10-19T16:01:16Z (branch creation) | v209-freeze | 2026-10-19T16:03:11Z (`astrocyte-v209-main01`, pushed and queued; Kaggle now shows its later start, 19:29:19Z) | Pushed to its own branch, cut from the then-current origin, so as not to push other work. The same freeze also sits on feature/documentation as 92c654bccc3c (first on origin 2026-10-19T20:30:11Z). The plan and smoke kernels (15:51) computed control quantities only. |
| v209 amendment 1 (plan read from kernel output) | v209 | same | b648949814dd | 2026-10-19T16:02:55Z | 2026-10-19T16:02:58Z | v209-freeze | as above | Infrastructure only (the embedded plan made the script too large for Kaggle). Copy on feature/documentation: 1a69cc293c2c. |
| Calibrated null for the mixture tests (Sec. 5.4) | v212 | `preregs/prereg_v212_mixture_null.md` | 6909b836bf6f | 2026-10-19T22:08:08Z | 2026-10-19T22:08:11Z (branch creation) | v212-freeze | 2026-10-19T22:08:34Z (`astrocyte-v212-cal-fourier`, stage 1) | Synthetic data only. Copy on feature/documentation: 05990e60393b. |
| v212 amendment 1 (calibration grid ×16) | v212 | same | eb87932aa99e | 2026-10-19T22:11:33Z | 2026-10-19T22:11:36Z | v212-freeze | 2026-10-19T22:11:41Z (`astrocyte-v212-cal-pair`, version 2) | Builder accepts a λ list; no rule changed. Copy on feature/documentation: 1091f5b64500. |

Exploratory analyses (v179–v181, v184, v185, v207) had no frozen pre-registration; their status is given in the
paper's Appendix C.

The development repository is private; commits are available to reviewers on request.
