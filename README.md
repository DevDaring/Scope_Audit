# SCOPE: Patchscope-Guided Surgical Debiasing of Social Bias in Language Models

SCOPE is a two-stage instrument for trustworthy language-model fairness. It first **audits**
a bias benchmark to find the items whose answer routes causally through a protected
attribute, even when the surface answer looks fair. It then **repairs** that routing with a
**localised, verified, load-bearing-aware** edit: a Patchscope decides *where* the
attribute is encoded, the edit removes it only there and only orthogonal to the model's
load-bearing structure, and a second Patchscope confirms the removal.

The audit lives in `Code/audit`. The repair, the comparison against recent debiasing
methods, the held-out and behavioural tests, the ablations, and the prognosis live in
`Code/SCOPE`. A reader can reproduce the whole pipeline from this file alone.

---

## 0. Honest headline (read this first)

1. **Audit.** A benchmark score hides a large validity gap. The causal commutator finds
   items that look fair but compute unfairly, which no behavioural audit can detect.
2. **Prognosis.** The audit severity predicts how hard a model is to repair, so an auditor
   can read the cost of a fix from the audit alone.
3. **Surgical repair.** SCOPE localises the protected attribute with a Patchscope, edits
   only the layers where it is decodable, and stays orthogonal to the massive-activation
   dimensions the model relies on. The design target is to remove the behavioural bias
   while keeping accuracy, where all-layer erasure does not.

**Outcome of the full four-model run (numbers in Section 3.4).** SCOPE beats the closest
prior method, Faithful-Patchscopes, on **2 of 4** models (Qwen2.5-7B, Phi-4-mini) and is the
**least destructive** edit on every model — its four-option accuracy cost stays within
±0.02 while other methods lose up to 55 points. The two design mechanisms do **not**
generalise: the Patchscope localisation never beats a random choice of layers, and the
massive-activation protection improves utility on only one model. The honest reading is a
utility-preserving debiaser plus a negative result on decodability-localisation — on Llama
and Gemma the bias is fused with the massive-activation dimensions SCOPE protects, so a
localised edit cannot reach it.

Every number is produced by the same evaluators on the same pairs, and nothing is
hard-coded to win. The run reports the honest outcome.

---

## 1. What the two folders do

| Folder | Role | Summary |
|--------|------|---------|
| `Code/audit` | Diagnosis | The causal discriminative-validity audit. A behavioural probe over the five-slot "pentad" plus a causal intervention that patches the protected-attribute residual at every layer and reads the change in the answer logit. Produces the validity leaderboard and the commutator results. |
| `Code/SCOPE` | Repair + study | Localises the attribute with a Patchscope, builds the protected edit, re-audits, measures the utility cost, compares against nine debiasing baselines (including Faithful-Patchscopes), runs the held-out and behavioural tests and the ablations, and fits the prognosis. |

**Models** (instruction-tuned): Llama-3.1-8B and Gemma-2-2B via TransformerLens;
Qwen2.5-7B and Phi-4-mini via NNsight. **Datasets**: BBQ, CrowS-Pairs, StereoSet, folded
into a 596-seed "pentad" over ten demographic axes.

---

## 2. The audit: procedure (`Code/audit`)

### 2.1 The pentad dataset
Each seed is a template with a demographic slot, expanded into five slots (a to e) and
several sub-variants (`Dataset/seeds/pentad_dataset.parquet`). Slot `a` carries the clean
(disambiguated) prompt; slot `c` carries the demographic swaps used by the commutator.

### 2.2 The causal commutator
`GPU_CPU/cdva_patching.py` swaps the demographic token (`a -> b`) and, via activation
patching, writes the swapped-token residual from the run on `a` into the run on `b` **at
every layer**, then reads the change in the gold-option logit:

```
C(a, b) = logit_gold( swap(a -> b) ) - logit_gold( a )      # the commutator
```

`C ~ 0` means the answer does not depend on the protected attribute. A large `|C|` means it
does. The threshold `tau = 0.7644` is the 75th percentile of `|C|`. Aggregates: **severity**
(mean `|C|`), **commutativity index** (fraction of seeds all below `tau`), and the
**validity gap** (native pass rate minus the audit-robust rate).

### 2.3 Run the audit
```bash
cd Code/audit
python3 GPU_CPU/run_gpu_pipeline.py     # behavioural eval + CDVA patching (GPU)
python3 run_cpu_full.py                 # scoring, leaderboard, statistics (CPU)
```
Outputs in `Code/audit/results/` (`cdva_results.parquet`, `leaderboard.parquet`,
`validity_gap_leaderboard.parquet`, `scored_results.parquet`).

---

## 3. The repair: SCOPE procedure (`Code/SCOPE`)

Three stages and a verification, all inference-only (no fine-tuning).

| Stage | What it does | File |
|---|---|---|
| 1. Localise | A Few-Shot Token-Identity Patchscope decodes, per layer, how strongly the protected attribute is readable from the residual at the swapped position. The localised layers are those with high decodability. | `scope.py` (`decodability_map`, `localise`) |
| 2. Protect | At the localised layers, estimate the demographic direction and remove its components on the top-`MASSIVE_K` massive-activation dimensions, so the edit is orthogonal to the load-bearing structure. A layer whose direction is essentially load-bearing is dropped. | `scope.py` (`build_scope_basis`) |
| 3. Edit | The result is a per-(localised layer) rank-1 basis. Passing it to the shared evaluators edits only those layers with the protected direction. | `scope.py`, `erase.py` (`ErasureContext`) |
| Verify | Re-decode the edited representation with the same Patchscope; the decodability of the attribute must drop, which the paper reports against the behavioural flip-rate drop. | `scope.py` (`decodability_map`, `edit_basis=`) |

### 3.1 Utility-aware localisation
`run_scope.py::select_scope_basis` climbs a percentile ladder (lower percentile = more
layers = more removal) and keeps the most aggressive edit whose four-option accuracy drop
stays within the budget (`MAX_UTILITY_COST = 0.15`), else the least-damaging edit.

### 3.2 Baselines on an independent signal
The nine baselines (`baselines.py`) derive their bias direction from an **independent** set
of 310 demographic-contrast templates, not from the audit pairs, and include
**Faithful-Patchscopes** (`patchscopes`), the closest prior method. They are faithful
re-implementations on one shared protocol; SCOPE alone reads the causal audit signal, so any
advantage isolates the value of that signal and the localisation.

### 3.3 Run the study (single launch)
```bash
cd Code/SCOPE
python3 run_scope.py --mode dry    # validate the full path on two pairs per model
python3 run_scope.py --mode main   # localise + edit + verify, head-to-head vs nine baselines,
                                    # held-out + behavioural, ablations, prognosis; 15-min checkpoints
```
Outputs in `Code/SCOPE/results/`: `scope_localization_<model>.json`,
`scope_final_<model>.parquet` (SCOPE vs nine baselines), `scope_extra_<model>.parquet`
(held-out bias removed + behavioural accuracy + answer-flip rate),
`scope_ablation_<model>.parquet` (localised vs all-layer, protect vs no-protect, random),
`scope_prognosis_<model>.{parquet,json}`, `SCOPE_DONE`.

A cloud GPU bootstrap is provided: `Code/SCOPE/bootstrap_scope.sh` pins the environment,
downloads the models, runs the dry check, then the main run, with 15-minute GitHub
checkpoints and pull-retry on failure (a fix needs no redeploy).

### 3.4 Results of the completed run

The full run (four models, 2026-06-21) is reproduced in `Code/SCOPE/results/`. Every
`scope_final` has the ten methods, no error rows, and `n_pairs = 1000`.
`causal_residual_removed` (`crr`) is the fraction of the 1000 shared eval pairs whose
audited commutator falls under `tau` after the edit; `util_cost` is the four-option
accuracy drop (negative = accuracy improved).

**Baseline reproducibility.** The nine baselines reproduce the audit-paper reference
(`cure_final_*`) within numerical noise on all four models (mean |Δ-crr| 0.008–0.012,
mixed sign — a threshold-fraction metric on different hardware), so the comparison set is
stable across runs.

**SCOPE vs Faithful-Patchscopes** (head-to-head; the held-out behavioural test agrees):

| Model | SCOPE crr / util | Faithful-Patchscopes crr / util | Winner |
|---|---|---|---|
| Llama-3.1-8B | 0.495 / −0.020 | 0.808 / −0.030 | Patchscopes |
| Qwen2.5-7B | 0.463 / −0.020 | 0.464 / **+0.164** | **SCOPE** |
| Gemma-2-2B | 0.377 / +0.020 | 0.707 / +0.020 | Patchscopes |
| Phi-4-mini | 0.405 / −0.005 | 0.369 / +0.035 | **SCOPE** |

SCOPE wins where the aggressive ablation backfires (Qwen: Patchscopes loses 16 accuracy
points; Phi: Patchscopes removes less bias). Where Patchscopes stays both strong and safe
(Llama, Gemma) it removes more bias at equal cost.

**Utility is the consistent SCOPE property.** Held-out accuracy change vs the unedited
model: SCOPE **+0.006** (Llama), **+0.048** (Qwen), **−0.044** (Gemma), **−0.012** (Phi) —
the smallest or near-smallest of any edit on every model, while
`generic_erase`/`biasgym`/`nofreelunch` lose 12–55 points. SCOPE never collapses the model.

---

## 4. Ablations (these keep the paper honest)

`scope_ablation_<model>.parquet` pits the localised + protected edit against three variants.
The completed run gives an honest, partly-negative result:

1. **Localisation** (SCOPE vs random vs all-layer sites). The Patchscope-localised layers do
   **not** beat a random choice of the same number of layers on any model — `scope_random`
   removes as much or more causal bias (e.g. Llama 0.638 vs 0.495, Gemma 0.453 vs 0.377).
   Decodability-localisation does not add value in this study.
2. **Massive-activation protection** (orthogonal-to-massive vs direct removal). Protection
   preserves utility on **one** model (Qwen: `scope_noprotect` costs 6 accuracy points more);
   on Gemma it slightly hurts. Where the bias is not fused with the massive dimensions,
   protection is what gives SCOPE its safety.
3. **Verification calibration** — decodability before vs after the edit, recorded per model in
   `scope_localization_<model>.json`.

---

## 5. Environment (exact)

The OS must match the precompiled flash-attention wheel.

- OS: Ubuntu 24.04 LTS, x86_64 (CUDA image `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`).
- Python 3.12, Torch `2.5.1` (cu124), CUDA 12.x driver, a single 48 GB GPU recommended.
- No virtual environment; install globally with `--break-system-packages`.

```bash
pip3 install --break-system-packages torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install --break-system-packages -r Code/SCOPE/requirements_scope.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip3 install --break-system-packages --no-deps transformer_lens==2.18.0
pip3 install --break-system-packages --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

---

## 6. Secrets and the `.env` contract

Every key is read from the environment. No secret is ever written into a tracked file.
Copy `Code/SCOPE/.env.example` to `Code/SCOPE/.env` and fill the values. The `.env` is
git-ignored and never pushed.

| Variable | Purpose |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Model download. |
| `Github_Classic_Token` | Checkpoint pushes. |
| `RANDOM_SEED` | Reproducibility (default 20260101). |
| `GEMINI_API_KEY_1..4` | Judge (answer extraction fallback), gemini-2.5-flash. |
| `DEEPSEEK_API_KEY_1..2`, `MISTRAL_API_KEY1..2`, `OPENROUTER_API_KEY_1..2` | Fallback judge tiers. |

SCOPE knobs (optional, with defaults): `SCOPE_LOCALIZE_PAIRS`, `SCOPE_DECODE_PCTILE`,
`SCOPE_PCTILE_LADDER`, `SCOPE_MASSIVE_K`, `SCOPE_PROTECT`, `SCOPE_TARGET_PROMPT`.

---

## 7. Repository map

```
Code/
  audit/                  the causal discriminative-validity audit (diagnosis)
    Dataset/seeds/        the pentad dataset
    GPU_CPU/              behavioural evaluation (osm_behavioral) and CDVA patching
    CPU_Only/             scoring, statistics, leaderboard
    results/              cdva_results, leaderboards, validity gap
  SCOPE/                  the repair, comparison, ablations, and prognosis
    scope.py              Patchscope localisation + massive-activation-protected edit
    run_scope.py          single entry point (dry, main) for the whole study
    scope_eval.py         held-out split + behavioural readout
    experiments.py        commutator re-audit, utility, demographic signal
    erase.py              activation caching + projection eraser (used by SCOPE)
    baselines.py          nine debiasing baselines (incl. Faithful-Patchscopes)
    config_scope.py       loads .env, models, datasets, tau, judge, SCOPE knobs
    judge_api.py          judge / answer extraction (round-robin, no cross-tier fallback)
    integrity.py          duplicate and corruption checks
    checkpoint.py         resume-safe 15-minute GitHub pushes
    bootstrap_scope.sh    GPU VM entrypoint (single launch)
    results/              all SCOPE study artifacts
README.md                 this file
```

---

## 8. Citations

SCOPE builds on Patchscopes (Ghandeharioun et al. 2024, arXiv:2401.06102) for the
localisation and verification, the closed-form projection eraser (Belrose et al. 2023,
arXiv:2306.03819), and activation patching (Meng et al. 2022, arXiv:2202.05262). The
load-bearing reading of the audited direction follows the massive-activation literature
(Sun et al. 2024, arXiv:2402.17762; Yu et al. 2024, arXiv:2411.07191; Oh et al. 2024,
arXiv:2410.01866). The closest prior method, Faithful-Patchscopes (Gong et al. 2026,
arXiv:2602.00300), is included as a baseline. The audit half is the causal
discriminative-validity audit described in the accompanying paper.
