# XAI + human-in-the-loop experiments (FraudBench-NG)

Code and results for "Explainable AI with Human-in-the-Loop Feedback for Real-Time Fraud Detection in Nigerian Inter-Bank Payment Systems".

Run from the repo root so `generate_dataset.py` and `data/nibss_synthetic_fraud.parquet` are found:

    cd experiments/xai_hitl
    PYTHONPATH=../.. python3 xai_stage1_models.py     # detectors, ADASYN ablation, tiers
    PYTHONPATH=../.. python3 xai_stage2_explain.py    # SHAP, faithfulness, stability, latency
    PYTHONPATH=../.. python3 xai_stage3_hitl.py       # feedback simulation (about 1 h)
    python3 hitl_summary.py && python3 make_xai_figs.py && python3 make_xai_figs2.py

Stage 2 and 3 need `results/deployed.joblib` from stage 1 (not committed). `results/` holds the JSON outputs used in the paper.
