"""Matplotlib chart builders, restyled to the navy/green/gold brand palette."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from components.theme import NAVY, GREEN, GOLD, TEXT_MUTED, BORDER

matplotlib_rc = {"font.family": "sans-serif"}

_NEUTRAL = "#AAB4C2"
_NEUTRAL_LIGHT = "#EDEFF2"
_RISK_UP = "#B3261E"
_RISK_DOWN = GREEN


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED)


def shap_bar(shap_vals, feat_names, title="SHAP Feature Attribution"):
    """Plot SHAP bar chart for a single transaction."""
    top_n = 8
    paired = sorted(zip(shap_vals, feat_names), key=lambda x: -abs(x[0]))[:top_n]
    vals = [p[0] for p in paired]
    names = [p[1] for p in paired]
    colours = [_RISK_UP if v > 0 else _RISK_DOWN for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colours, edgecolor="black", linewidth=0.6, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_xlabel("SHAP Value (red = increases fraud score, green = decreases)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY)
    ax.invert_yaxis()
    _style_axes(ax)
    fig.tight_layout()
    return fig


def performance_bars(res, models4):
    """3-panel comparison chart: PR-AUC / Recall / F1 across the four configs."""
    configs = ["B1: RF\nAlone", "B2: IF\nAlone", "B3: RF+\nADASYN", "Full\nHybrid"]
    clrs = [_NEUTRAL_LIGHT, _NEUTRAL, _NEUTRAL_LIGHT, GOLD]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Across Four Configurations", fontsize=13, fontweight="bold", color=NAVY)
    for ax, (metric, key, title) in zip(axes, [
        ("PR-AUC", "PR_AUC", "PR-AUC (Primary)"),
        ("Recall", "Recall", "Recall"),
        ("F1", "F1", "F1-Score"),
    ]):
        vals = [res[m][key] for m in models4]
        bars = ax.bar(configs, vals, color=clrs, edgecolor=NAVY, linewidth=1.2, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=NAVY)
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)
        ax.set_ylim(0, min(1.0, max(vals) * 1.35))
        ax.set_ylabel("Score", fontsize=11)
        ax.tick_params(labelsize=11)
        _style_axes(ax)
    plt.tight_layout()
    return fig


def feature_importance_barh(fi_data, top_n=15):
    """Horizontal bar chart of the top-N feature importances."""
    names_fi = [f[0] for f in fi_data[:top_n]]
    vals_fi = [f[1] for f in fi_data[:top_n]]
    cmap = {
        "amount": NAVY, "amount_log": NAVY, "amount_vs_mean_ratio": NAVY, "amount_sum_24h": NAVY,
        "velocity_score": GREEN,
        "month_sin": GOLD, "month_cos": GOLD, "day_cos": GOLD, "is_peak_hour": GOLD,
        "channel_Mobile": _NEUTRAL,
    }
    bar_c = [cmap.get(n, _NEUTRAL) for n in names_fi]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names_fi))
    ax.barh(y_pos, vals_fi, color=bar_c, edgecolor=NAVY, linewidth=0.8, height=0.65)
    for i, val in enumerate(vals_fi):
        ax.text(val + 0.0008, i, f"{val:.4f}", va="center", fontsize=10, color=NAVY)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_fi, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Top 15 Feature Importances (Full Hybrid RF Component)", fontsize=12, fontweight="bold", color=NAVY)
    _style_axes(ax)
    handles = [
        Patch(fc=NAVY, ec=NAVY, label="Amount features"),
        Patch(fc=GREEN, ec=NAVY, label="Composite risk"),
        Patch(fc=GOLD, ec=NAVY, label="Temporal features"),
        Patch(fc=_NEUTRAL, ec=NAVY, label="Channel features"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower right")
    plt.tight_layout()
    return fig


def hitl_charts(hitl):
    """3-panel HITL chart: TP blocked / recall / PR-AUC across feedback cycles."""
    cycles = [h["cycle"] for h in hitl]
    tp_block = [h["tp_block"] for h in hitl]
    recall_bl = [h["recall_block"] for h in hitl]
    pr_auc_vals = [h["pr_auc"] for h in hitl]

    clrs4 = [_NEUTRAL_LIGHT, _NEUTRAL, _NEUTRAL, GOLD]
    xlabs = ["Cycle 0\n(Baseline)", "Cycle 1", "Cycle 2", "Cycle 3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("HITL Feedback Results - Three Retraining Cycles\nFixed threshold θ = 0.70",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax1, ax2, ax3 = axes

    bars = ax1.bar(cycles, tp_block, color=clrs4, edgecolor=NAVY, linewidth=1.3, width=0.65)
    for bar, val in zip(bars, tp_block):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(val),
                 ha="center", va="bottom", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xticks(cycles)
    ax1.set_xticklabels(xlabs, fontsize=11)
    ax1.set_ylabel("Fraud Cases Auto-Blocked", fontsize=12)
    ax1.set_ylim(0, 280)
    ax1.set_title("Fraud Auto-Blocked\n(No Analyst Review Required)", fontsize=12, fontweight="bold", color=NAVY)
    _style_axes(ax1)

    ax2.plot(cycles, recall_bl, color=NAVY, marker="o", lw=2.5, ms=9, mfc=GOLD, mec=NAVY)
    for x, y in zip(cycles, recall_bl):
        ax2.text(x + 0.06, y + 0.012, f"{y:.4f}", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xticks(cycles)
    ax2.set_xticklabels(xlabs, fontsize=11)
    ax2.set_ylabel("Recall at BLOCK Tier", fontsize=12)
    ax2.set_ylim(0.35, 0.80)
    ax2.set_title("Recall Improvement at BLOCK Tier", fontsize=12, fontweight="bold", color=NAVY)
    _style_axes(ax2)

    ax3.plot(cycles, pr_auc_vals, color=NAVY, marker="o", lw=2.5, ms=9, mfc=GOLD, mec=NAVY)
    for x, y in zip(cycles, pr_auc_vals):
        ax3.text(x + 0.06, y + 0.004, f"{y:.4f}", fontsize=11, fontweight="bold", color=NAVY)
    ax3.set_xticks(cycles)
    ax3.set_xticklabels(xlabs, fontsize=11)
    ax3.set_ylabel("PR-AUC", fontsize=12)
    ax3.set_ylim(0.75, 0.90)
    ax3.set_title("PR-AUC Across Feedback Cycles", fontsize=12, fontweight="bold", color=NAVY)
    _style_axes(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
