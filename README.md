# XAI + Human-in-the-Loop Fraud Detection (FraudBench-NG)

Code and results for the paper "Explainable AI with Human-in-the-Loop Feedback for Real-Time Fraud Detection in Nigerian Inter-Bank Payment Systems".

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Experiment scripts: shared utilities (`xai_common.py`), the three experiment stages, summary and figure scripts |
| `results/` | JSON outputs and run logs used in the paper |
| `figures/` | Figures generated for the paper |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/xai_stage1_models.py` | Detectors, ADASYN ablation, tiers |
| 2 | `src/xai_stage2_explain.py` | SHAP, faithfulness, stability, latency |
| 3 | `src/xai_stage3_hitl.py` | Analyst feedback simulation (about 1 h) |
| 4 | `src/hitl_summary.py` | Summarises stage 3 results |
| 5 | `src/make_xai_figs.py`, `src/make_xai_figs2.py`, `src/make_hitl_fig.py`, `src/make_xai_arch.py`, `src/make_xai_fig_scores.py` | Paper figures |

## Running

Install the dependencies, then run every script from the repository root so `results/` and `figures/` resolve:

    pip install -r requirements.txt
    python src/xai_stage1_models.py
    python src/xai_stage2_explain.py
    python src/xai_stage3_hitl.py
    python src/hitl_summary.py
    python src/make_xai_figs.py
    python src/make_xai_figs2.py

Stages 1 and 2 import `generate_dataset` and load `data/nibss_synthetic_fraud.parquet`. Neither is included in this repository, so add them at the repository root before running. Stages 2 and 3 need `results/deployed.joblib` from stage 1, which is not committed.
