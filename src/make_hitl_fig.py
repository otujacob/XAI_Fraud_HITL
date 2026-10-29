import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.load(open('results/xai_hitl_summary.json')); st = {'alert': ('#2A9D8F', 'o', 'Alert review (top-300)'), 'random': ('#8A8F98', 's', 'Random labels (300)'), 'noisy': ('#E76F51', '^', 'Noisy analysts'), 'oracle': ('#8D5FD3', 'D', 'Oracle')}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for a, (k, t) in zip(ax, (('pr_auc', '(a) PR-AUC'), ('rec_fpr1', '(b) Recall at 1% FPR'), ('rec_fpr01', '(c) Recall at 0.1% FPR'))):
    for arm, (c, mk, lb) in st.items():
        v = np.array(S[f'N300_{k}'][arm]); x = np.arange(4) + {'alert': -.06, 'random': -.02, 'noisy': .02, 'oracle': .06}[arm]
        a.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, marker=mk, ms=3.2, lw=1, capsize=1.5, label=lb)
    a.set_xticks(range(4)); a.set_xticklabels(['0', '1', '2', '3']); a.set_xlabel('Feedback cycle'); a.set_title(t, fontsize=8)
ax[0].legend(frameon=False, fontsize=6.2, loc='upper left')
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_hitl.png', dpi=300); print('ok')
