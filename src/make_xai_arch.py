"""Architecture figure for the XAI + HITL paper (double-column width, readable fonts)."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300})
TEAL, CORAL, PURP, GOLD, BLUE, GREY = '#2A9D8F', '#E76F51', '#8D5FD3', '#E9C46A', '#4C78A8', '#8A8F98'

def box(ax, x, y, w, h, text, fc, fs=7.6, tc='black', lw=.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec='#333', lw=lw, alpha=.95))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, linespacing=1.25)
def arrow(ax, p, q, c='#333', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1, color=c, connectionstyle=f'arc3,rad={rad}', linestyle=ls))

fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.set_xlim(0, 24); ax.set_ylim(0, 10.3); ax.axis('off')
box(ax, 0.2, 6.3, 4.3, 2.6, 'Transaction\n+ causal history\nfeatures (48)', '#BFD4EA')
box(ax, 5.3, 6.3, 4.1, 2.6, 'Stage 1\nIsolation Forest\nanomaly score', GOLD)
box(ax, 10.2, 6.3, 4.6, 2.6, 'Stage 2\nXGBoost\nclassifier', TEAL)
box(ax, 15.6, 6.3, 3.6, 2.6, 'Risk score\n$\\hat{P}$', '#E8E8E8')
box(ax, 20.0, 8.0, 3.8, 1.4, 'APPROVE', '#9BD3A8')
box(ax, 20.0, 6.3, 3.8, 1.4, 'BLOCK', CORAL, tc='white')
box(ax, 20.0, 4.6, 3.8, 1.4, 'REVIEW', GOLD)
for a, b in ((4.5, 5.3), (9.4, 10.2), (14.8, 15.6)): arrow(ax, (a, 7.6), (b, 7.6))
arrow(ax, (19.2, 8.3), (20.0, 8.7)); arrow(ax, (19.2, 7.6), (20.0, 7.0)); arrow(ax, (19.2, 6.9), (20.0, 5.3))
box(ax, 10.2, 2.35, 5.2, 2.5, 'Stage 3\nSHAP TreeExplainer\n+ amount counterfactual', PURP, tc='white')
arrow(ax, (12.5, 6.3), (12.5, 4.85), '#333')
box(ax, 16.4, 2.3, 4.0, 2.1, 'Analyst dashboard\n(score, reasons)', '#F6D5C9')
arrow(ax, (15.4, 3.5), (16.4, 3.5)); arrow(ax, (21.9, 4.6), (20.4, 3.9), rad=-.2)
box(ax, 5.3, 0.2, 5.0, 1.6, 'Labelled decisions\n(confirm / false alarm)', '#E8E8E8')
box(ax, 0.2, 2.35, 4.3, 2.5, 'Periodic retraining\n(feedback cycle)', BLUE, tc='white')
arrow(ax, (18.0, 2.3), (10.3, 1.0), CORAL, rad=.15); arrow(ax, (5.3, 1.0), (2.35, 2.35), CORAL, rad=.15)
arrow(ax, (4.5, 4.2), (10.2, 6.6), BLUE, rad=-.2, ls='--')
ax.text(23.8, 0.3, 'solid: scoring and review path   dashed: updated model', ha='right', fontsize=6.6, color='#444')
plt.tight_layout(pad=.3); plt.savefig('figures/xai_fig_arch.png', dpi=300); print('ok')
