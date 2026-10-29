import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
m = json.load(open('results/xai_models.json')); e = json.load(open('results/xai_explain.json')); z = np.load('results/xai_explain_arrays.npz', allow_pickle=True)
yt = np.load('results/xai_yt.npy')
# Fig: performance
order = ['IF alone', 'RF', 'RF+ADASYN', 'XGB', 'XGB+IF', 'XGB+IF+ADASYN', 'Hybrid ensemble']
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [1.15, 1, 1]})
for k, c in (('RF', '#E76F51'), ('XGB+IF', '#2A9D8F'), ('Hybrid ensemble', '#8D5FD3'), ('IF alone', '#8A8F98')):
    p = np.load('results/xai_p_' + k.replace('+', '_').replace(' ', '_') + '.npy'); pr, rc, _ = precision_recall_curve(yt, p)
    ax[0].plot(rc, pr, color=c, lw=1.2, label=k)
ax[0].axhline(yt.mean(), color='k', ls=':', lw=.8, label='No skill'); ax[0].set_yscale('log'); ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title('(a) Precision-recall', fontsize=8)
for j, (key, ttl) in enumerate((('pr_auc', '(b) PR-AUC'), ('rec_fpr1', '(c) Recall at 1% FPR'))):
    a = ax[j + 1]; v = [m['configs'][k]['test'][key] for k in order]; lo = [m['configs'][k]['ci'][key][0] for k in order]; hi = [m['configs'][k]['ci'][key][1] for k in order]
    yy = np.arange(len(order))[::-1]; a.errorbar(v, yy, xerr=[np.array(v) - lo, np.array(hi) - v], fmt='o', ms=3.5, color='#2A9D8F', ecolor='#555', capsize=2, lw=1)
    a.set_yticks(yy); a.set_yticklabels(order if j == 0 else [''] * len(order), fontsize=7); a.set_title(ttl, fontsize=8)
    if key == 'pr_auc': a.axvline(yt.mean(), color='k', ls=':', lw=.8)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_perf.png', dpi=300); plt.close()
# Fig: explanations
lab = {'amount_vs_mean_ratio': 'Amount / customer mean', 'tx_count_total': 'Prior tx count', 'tx_count_24h': 'Tx in last 24 h', 'amount_mean_total': 'Customer mean amount', 'channel_ECOM': 'Channel: e-commerce', 'channel_IB': 'Channel: internet banking', 'amount_std_total': 'Customer amount std.', 'amount_mean_7d': '7-day mean amount'}
tf = list(e['top_features'].items())[:8][::-1]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax[0].barh([lab.get(k, k) for k, _ in tf], [v for _, v in tf], color='#2A9D8F'); ax[0].set_xlabel('Mean |SHAP| over alerts (log-odds)'); ax[0].set_title('(a) Top features', fontsize=8)
ks = [1, 3, 5, 10]; f = e['faithfulness']
ax[1].plot(ks, [f[str(k)]['mean_drop_shap'] for k in ks], 'o-', color='#8D5FD3', label='Top-$k$ by SHAP'); ax[1].plot(ks, [f[str(k)]['mean_drop_random'] for k in ks], 's--', color='#8A8F98', label='Random $k$')
ax[1].set_xlabel('Features neutralised ($k$)'); ax[1].set_ylabel('Mean score drop'); ax[1].set_xticks(ks); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title('(b) Deletion test', fontsize=8)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_shap.png', dpi=300); print('ok')
