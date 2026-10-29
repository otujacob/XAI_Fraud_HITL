import numpy as np, joblib, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
D = joblib.load('results/deployed.joblib'); p = np.load('results/xai_p_XGB_IF.npy'); yt = np.load('results/xai_yt.npy')
fig, ax = plt.subplots(figsize=(3.5, 2.3)); b = np.linspace(0, 1, 41)
ax.hist(p[yt == 0], b, color='#8A8F98', alpha=.8, label='Legitimate', weights=np.full((yt == 0).sum(), 1 / (yt == 0).sum()))
ax.hist(p[yt == 1], b, color='#E76F51', alpha=.7, label='Fraud', weights=np.full((yt == 1).sum(), 1 / (yt == 1).sum()))
ax.set_yscale('log'); ax.axvline(D['thr_review'], color='#E9C46A', lw=1.4, label='REVIEW'); ax.axvline(D['thr_block'], color='#333', lw=1.4, ls='--', label='BLOCK')
ax.set_xlabel('Risk score'); ax.set_ylabel('Share of class'); ax.legend(frameon=False, fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_scores.png', dpi=300); print('ok')
