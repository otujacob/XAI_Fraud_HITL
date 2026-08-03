import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
import pickle, json, os, warnings, time
warnings.filterwarnings('ignore')

from components.theme import inject_css
from components.sidebar import render_sidebar
from components.topbar import render_topbar
from components.kpi import kpi_row
from components.badges import decision_badge_html
from components.charts import shap_bar, performance_bars, feature_importance_barh, hitl_charts
from components.layout import page_header, hero_header, section_title, section_card
from components.timeline import timeline
from components.footer import render_footer

# Page config
st.set_page_config(
    page_title="XAI Fraud Detection - Nigerian Inter-Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_css()

# Constants
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
CAT_FEATS  = ['channel','merchant_category','bank','location','age_group']
DROP_COLS  = ['transaction_id','customer_id','timestamp','is_fraud','fraud_technique']
THETA_BLOCK  = 0.70
THETA_REVIEW = 0.40

# Load models (cached)
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    with open(f'{MODEL_DIR}/rf_model.pkl','rb')   as f: rf  = pickle.load(f)
    with open(f'{MODEL_DIR}/xgb_model.pkl','rb')  as f: xgb = pickle.load(f)
    with open(f'{MODEL_DIR}/scaler.pkl','rb')      as f: sc  = pickle.load(f)
    with open(f'{MODEL_DIR}/iso_forest.pkl','rb')  as f: iso = pickle.load(f)
    with open(f'{MODEL_DIR}/feature_names.json')   as f: fnames = json.load(f)
    with open(f'{MODEL_DIR}/results.json')         as f: res = json.load(f)
    return rf, xgb, sc, iso, fnames, res

rf_model, xgb_model, scaler, iso_forest, FEATURE_NAMES, PRECOMPUTED = load_models()

# Helpers
def preprocess(df_input):
    """Preprocess a transaction dataframe for scoring."""
    feat_cols = [c for c in df_input.columns if c not in DROP_COLS]
    enc = pd.get_dummies(df_input[feat_cols], columns=[c for c in CAT_FEATS if c in feat_cols])
    # Align to training feature set (all 69 features)
    base_names = [n for n in FEATURE_NAMES if n != 'if_anomaly_score']
    for col in base_names:
        if col not in enc.columns:
            enc[col] = 0
    enc = enc[[c for c in base_names if c in enc.columns]]
    for col in base_names:
        if col not in enc.columns:
            enc[col] = 0
    enc = enc[base_names]
    X_sc = scaler.transform(enc.values.astype(float))
    if_sc = iso_forest.score_samples(X_sc).reshape(-1,1)
    X_aug = np.hstack([X_sc, if_sc])
    return X_aug

def score_transactions(X_aug):
    p_rf  = rf_model.predict_proba(X_aug)[:,1]
    p_xgb = xgb_model.predict_proba(X_aug)[:,1]
    p_ens = (p_rf + p_xgb) / 2
    return p_rf, p_xgb, p_ens

def decision(prob):
    if prob >= THETA_BLOCK:  return 'BLOCK',  '🔴'
    if prob >= THETA_REVIEW: return 'REVIEW', '🟡'
    return 'APPROVE', '🟢'

# Sidebar
page = render_sidebar()

# Top bar (university logos, persistent across pages)
render_topbar()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    hero_header(
        badges=[("COM752", "pill-navy"), ("DISSERTATION", "pill-green-outline")],
        title="XAI Fraud Detection for Nigerian Inter-Bank Payments",
        quote='"Explainable AI with Human-in-the-Loop Feedback for Real-Time Fraud Detection '
              'in Nigerian Inter-Bank Payment Systems"',
        caption="COM 752 MSc Dissertation, Wrexham University, 2025/2026",
    )

    s = PRECOMPUTED['summary']
    kpi_row([
        {"label": "Total Transactions", "value": f"{s['total_transactions']:,}", "delta": "Processed",            "icon": "🏦", "accent": "green"},
        {"label": "Fraud Cases",        "value": f"{s['fraud_cases']:,}",        "delta": "Detected",              "icon": "🛡️", "accent": "blue"},
        {"label": "Class Imbalance",    "value": f"{s['class_ratio']}:1",        "delta": "Fraud : Legitimate",    "icon": "⚖️", "accent": "purple"},
        {"label": "P95 Latency",        "value": f"~{s['latency_p95']}ms",       "delta": "End-to-End",            "icon": "⚡", "accent": "gold"},
    ])

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with section_card():
            section_title("📦", "Four-Component Framework")
            timeline([
                {"title": "Stage 1 - Isolation Forest", "desc": "Unsupervised anomaly scoring → Feature 69", "icon": "📈"},
                {"title": "Stage 2 - Hybrid Classifier", "desc": "XGBoost + Random Forest trained on ADASYN-balanced data", "icon": "🕸️"},
                {"title": "SHAP TreeExplainer", "desc": "Real-time natural language explanations + counterfactuals", "icon": "💬"},
                {"title": "Human-in-the-Loop", "desc": "Analyst feedback → iterative model retraining", "icon": "🧑‍💼"},
            ])
            st.markdown(
                '<div class="framework-strip">Framework: IF → XGBoost + RF → ADASYN → SHAP</div>',
                unsafe_allow_html=True,
            )
    with c2:
        with section_card():
            section_title("🏆", "Key Results")
            st.markdown("""
| Metric | Value |
|--------|-------|
| Best PR-AUC (XGBoost) | **0.8037** |
| Best Recall (XGBoost) | **0.6646** |
| False Positives (BLOCK) | **0** |
| SHAP Latency (P95) | **~340ms** |
| H1 (SHAP within 500ms) | ✅ Confirmed |
| H2 (HITL recall gain) | +25.9pp |
| H3 (PR-AUC > 0.91) | ❌ Not confirmed |
""")

    with section_card():
        col_l, col_r = st.columns([4, 1])
        with col_l:
            section_title("🛡️", "Nigerian Fraud Ecology")
            st.markdown(
                '<p class="collapsible-caption">Understanding the fraud landscape across Nigerian inter-bank payment systems.</p>',
                unsafe_allow_html=True,
            )
        with col_r:
            if "show_ecology" not in st.session_state:
                st.session_state.show_ecology = False
            label = "Hide Ecology ↑" if st.session_state.show_ecology else "View Ecology →"
            if st.button(label, key="toggle_ecology", use_container_width=True):
                st.session_state.show_ecology = not st.session_state.show_ecology
                st.rerun()
        if st.session_state.show_ecology:
            st.write("")
            kpi_row([
                {"label": "Social Engineering", "value": "64.3%", "delta": "dominant fraud type",   "icon": "🎭", "accent": "gold"},
                {"label": "Mobile Banking",     "value": "49.9%", "delta": "highest fraud channel", "icon": "📱", "accent": "green"},
                {"label": "Amount Features",    "value": "69.1%", "delta": "of SHAP attribution",   "icon": "💰", "accent": "blue"},
            ])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
    page_header("📊", "Model Performance Results",
                "Test Set: Nov–Dec 2023 &middot; 185,658 transactions &middot; 328 fraud cases")

    res = PRECOMPUTED['results']
    rows = []
    for model, metrics in res.items():
        rows.append({'Model': model, **metrics})
    df_res = pd.DataFrame(rows)

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: #E3F3EE' if v else '' for v in is_max]

    with section_card():
        section_title("📋", "Table 4.1: Complete Performance Comparison")
        st.dataframe(
            df_res.style
                .apply(highlight_best, subset=['PR_AUC'])
                .apply(highlight_best, subset=['Recall'])
                .format({'Precision':'{:.4f}','Recall':'{:.4f}','F1':'{:.4f}',
                         'AUC_ROC':'{:.4f}','PR_AUC':'{:.4f}','MCC':'{:.4f}'}),
            use_container_width=True, height=280
        )
        st.caption("Green highlight = best value. PR-AUC is the primary metric.")

    with section_card():
        section_title("📈", "Performance Comparison - Primary Metrics")
        models4 = list(res.keys())[:4]
        fig = performance_bars(res, models4)
        st.pyplot(fig)

    with section_card():
        section_title("🌲", "Feature Importance - Top 15 Features")
        fig2 = feature_importance_barh(PRECOMPUTED['feat_imp'])
        st.pyplot(fig2)
        st.info("Amount-based features account for **69.1%** of mean absolute SHAP attribution - confirming the social engineering fraud signature.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explanations":
    page_header("🔍", "SHAP Explanation Examples",
                "Three real transactions from the test set, showing how the SHAP explanation module works at the moment of the fraud decision.")

    cases = PRECOMPUTED['cases']
    tabs  = st.tabs(["Case A - High Risk (FRAUD)", "Case B - Medium Risk (LEGIT)", "Case C - Low Risk (LEGIT)"])

    for tab, key in zip(tabs, ['A', 'B', 'C']):
        c = cases[key]
        with tab:
            with section_card():
                d_str, _ = decision(c['prob_rf'])
                kpi_row([
                    {"label": "True Label",      "value": "FRAUD" if c['true_y']==1 else "LEGIT", "icon": "🏷️", "accent": "navy"},
                    {"label": "RF Probability",  "value": f"{c['prob_rf']:.4f}",  "icon": "🌲", "accent": "green"},
                    {"label": "XGB Probability", "value": f"{c['prob_xgb']:.4f}", "icon": "⚡", "accent": "gold"},
                    {"label": "Decision",        "value": decision_badge_html(d_str), "icon": "🚦", "accent": "red" if d_str=="BLOCK" else ("gold" if d_str=="REVIEW" else "green")},
                ])

                st.write("")
                kpi_row([
                    {"label": "Amount",  "value": f"NGN {c['amount']:,.2f}", "icon": "💵", "accent": "navy"},
                    {"label": "Channel", "value": c['channel'],              "icon": "📡", "accent": "blue"},
                    {"label": "Hour",    "value": f"{c['hour']:02d}:00",     "icon": "🕒", "accent": "gold"},
                    {"label": "Bank",    "value": c['bank'],                 "icon": "🏦", "accent": "purple"},
                ])

                st.markdown("**SHAP Attribution - Top 5 Features:**")
                shap_df = pd.DataFrame(c['shap'], columns=['Feature', 'SHAP Value'])
                shap_df['Direction'] = shap_df['SHAP Value'].apply(
                    lambda x: '🔴 Increases fraud score' if x > 0 else '🟢 Decreases fraud score')
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

                # Plot SHAP bar
                sv   = [s[1] for s in c['shap']]
                fn   = [s[0] for s in c['shap']]
                fig  = shap_bar(sv, fn, f"Case {key} - SHAP Attribution")
                st.pyplot(fig)

                # Natural language explanation
                top = c['shap'][0]
                direction = "elevated above" if top[1] > 0 else "below"
                if c['true_y'] == 1:
                    nl = (f"**Explanation:** This transaction was flagged because "
                          f"**{top[0]}** is the primary driver (SHAP = {top[1]:+.4f}), "
                          f"pushing the fraud probability to {c['prob_rf']:.4f}. "
                          f"The transaction amount of NGN {c['amount']:,.2f} "
                          f"is {direction} the customer's typical range.")
                    cf_thresh = round(c['amount'] * 0.35 / 1000) * 1000
                    cf = f"**Counterfactual:** This transaction would NOT be flagged if the amount were below approximately NGN {cf_thresh:,.0f}."
                else:
                    nl = (f"**Explanation:** This transaction was approved because "
                          f"**{top[0]}** (SHAP = {top[1]:+.4f}) reduces the fraud score. "
                          f"The amount of NGN {c['amount']:,.2f} is within normal range.")
                    cf = "**Counterfactual:** N/A - transaction is already approved."
                st.info(nl)
                st.success(cf)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: HITL FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "HITL Feedback":
    page_header("🔄", "Human-in-the-Loop Feedback Simulation",
                "Three analyst feedback cycles - showing how the model improves as analysts confirm fraud cases from the review queue.")

    hitl = PRECOMPUTED['hitl']

    with section_card():
        section_title("📋", "Table 4.2: HITL Simulation Results (θ = 0.70)")
        df_hitl = pd.DataFrame(hitl)
        df_hitl['Gain in TP'] = df_hitl['tp_block'] - df_hitl['tp_block'].iloc[0]
        df_hitl['Gain in TP'] = df_hitl['Gain in TP'].apply(lambda x: f'+{x}' if x > 0 else '-')
        st.dataframe(df_hitl[['cycle','tp_block','Gain in TP','fp_block',
                                'tp_review','recall_block','pr_auc']].rename(columns={
            'cycle':'Cycle','tp_block':'TP Blocked','fp_block':'FP Blocked',
            'tp_review':'TP in Review','recall_block':'Recall (Block)','pr_auc':'PR-AUC'
        }), use_container_width=True, hide_index=True)

    with section_card():
        fig = hitl_charts(hitl)
        st.pyplot(fig)

    with section_card():
        section_title("🏆", "Key Finding")
        kpi_row([
            {"label": "Recall Improvement",    "value": "+25.9pp",   "delta": "0.4451 → 0.7043", "icon": "📈", "accent": "green"},
            {"label": "Extra Fraud Blocked",   "value": "+85 cases", "delta": "146 → 231",       "icon": "🚫", "accent": "gold"},
            {"label": "New False Positives",   "value": "0",         "delta": "precision maintained", "icon": "✅", "accent": "blue"},
        ])
        st.info("**H2 reframing:** The baseline already achieves zero false positives before any feedback. HITL instead improves block-tier recall by 25.9 percentage points across three cycles with zero new false positives.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: LIVE SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Scoring":
    page_header("⚡", "Live Transaction Scoring",
                "Score new transactions through the full framework pipeline - Isolation Forest → XGBoost + RF → Decision.")

    decision_legend = " &nbsp; ".join([
        decision_badge_html("BLOCK"), decision_badge_html("REVIEW"), decision_badge_html("APPROVE")
    ])

    tab1, tab2 = st.tabs(["📁 Upload CSV", "🎯 Demo Transactions"])

    with tab1:
        with section_card():
            st.markdown("Upload a transaction CSV with the same column structure as the NIBSS dataset.")
            uploaded = st.file_uploader("Upload transaction CSV", type=['csv'])

            if uploaded:
                df_up = pd.read_csv(uploaded, low_memory=False)
                st.success(f"Loaded {len(df_up):,} transactions")

                if st.button("Score Transactions", type="primary"):
                    with st.spinner("Running pipeline: Standardise → IF Score → Classify → Explain..."):
                        try:
                            t0 = time.perf_counter()
                            X_aug = preprocess(df_up)
                            p_rf, p_xgb, p_ens = score_transactions(X_aug)
                            elapsed = (time.perf_counter() - t0) * 1000

                            decisions = [decision(p)[0] for p in p_ens]
                            df_up['RF_Prob']       = p_rf.round(4)
                            df_up['XGB_Prob']      = p_xgb.round(4)
                            df_up['Ensemble_Prob'] = p_ens.round(4)
                            df_up['Decision']      = decisions

                            n = len(df_up)
                            kpi_row([
                                {"label": "Total Scored", "value": f"{n:,}", "icon": "📥", "accent": "navy"},
                                {"label": "BLOCK",   "value": f"{(df_up['Decision']=='BLOCK').sum()}",   "icon": "🔴", "accent": "red"},
                                {"label": "REVIEW",  "value": f"{(df_up['Decision']=='REVIEW').sum()}",  "icon": "🟡", "accent": "gold"},
                                {"label": "APPROVE", "value": f"{(df_up['Decision']=='APPROVE').sum()}", "icon": "🟢", "accent": "green"},
                            ])
                            st.caption(f"Total time: {elapsed:.0f}ms · ~{elapsed/n:.1f}ms per transaction")

                            st.markdown(decision_legend, unsafe_allow_html=True)
                            st.dataframe(df_up[['amount','channel','Ensemble_Prob','Decision']].head(50),
                                         use_container_width=True, hide_index=True)

                            # SHAP for highest risk transaction
                            top_idx = p_ens.argmax()
                            st.subheader(f"SHAP Explanation - Highest Risk Transaction (prob={p_ens[top_idx]:.4f})")
                            try:
                                import shap
                                expl = shap.TreeExplainer(rf_model)
                                sv   = expl.shap_values(X_aug[top_idx:top_idx+1])
                                sv_f = sv[0,:,1] if sv.ndim==3 else sv[1][0]
                                fig  = shap_bar(sv_f, FEATURE_NAMES, "Highest Risk Transaction - SHAP Attribution")
                                st.pyplot(fig)
                            except Exception as e:
                                st.warning(f"SHAP computation skipped: {e}")
                        except Exception as e:
                            st.error(f"Scoring error: {e}")
                            st.info("Make sure your CSV has the same columns as the NIBSS dataset.")

    with tab2:
        with section_card():
            st.markdown("Score the built-in demo dataset (528 transactions: 328 fraud + 200 legitimate from the test set).")
            demo_path = os.path.join(DATA_DIR, 'demo_transactions.csv')

            if st.button("Run Demo Scoring", type="primary"):
                df_demo = pd.read_csv(demo_path, low_memory=False)
                with st.spinner("Scoring 528 demo transactions..."):
                    t0 = time.perf_counter()
                    X_aug = preprocess(df_demo)
                    p_rf, p_xgb, p_ens = score_transactions(X_aug)
                    elapsed = (time.perf_counter() - t0) * 1000

                    y_true = df_demo['is_fraud'].values if 'is_fraud' in df_demo.columns else None
                    decisions = [decision(p)[0] for p in p_ens]
                    df_demo['RF_Prob']       = p_rf.round(4)
                    df_demo['Ensemble_Prob'] = p_ens.round(4)
                    df_demo['Decision']      = decisions

                    kpi_row([
                        {"label": "Total Scored", "value": f"{len(df_demo):,}", "icon": "📥", "accent": "navy"},
                        {"label": "BLOCK",   "value": f"{(df_demo['Decision']=='BLOCK').sum()}",   "icon": "🔴", "accent": "red"},
                        {"label": "REVIEW",  "value": f"{(df_demo['Decision']=='REVIEW').sum()}",  "icon": "🟡", "accent": "gold"},
                        {"label": "APPROVE", "value": f"{(df_demo['Decision']=='APPROVE').sum()}", "icon": "🟢", "accent": "green"},
                    ])
                    st.caption(f"Scoring time: {elapsed:.0f}ms total · ~{elapsed/len(df_demo):.1f}ms/tx")

                    if y_true is not None:
                        from sklearn.metrics import average_precision_score
                        prauc = average_precision_score(y_true, p_ens)
                        st.metric("Live PR-AUC", f"{prauc:.4f}", "computed on demo set")

                    st.markdown(decision_legend, unsafe_allow_html=True)
                    st.dataframe(
                        df_demo[['amount','channel','bank','RF_Prob','Ensemble_Prob','Decision']]
                        .sort_values('Ensemble_Prob', ascending=False).head(30),
                        use_container_width=True, hide_index=True
                    )

# Footer
render_footer()
