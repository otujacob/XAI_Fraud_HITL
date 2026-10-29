import json, numpy as np, joblib, shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import xai_common as C
plt.rcParams.update({'font.size': 8, 'figure.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
m = json.load(open('results/xai_models.json')); e = json.load(open('results/xai_explain.json')); S = json.load(open('results/xai_hitl_summary.json'))
# --- ADASYN fig
ab = m['adasyn_ablation']; keys = ['None', '0.01', '0.02', '0.05', '0.2']; lab = ['none', '0.01', '0.02', '0.05', '0.20']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={'width_ratios': [1, 1.15]})
x = np.arange(5); ax[0].bar(x - .18, [ab[k]['val_pr_auc'] for k in keys], .36, color='#4C78A8', label='Validation'); ax[0].bar(x + .18, [ab[k]['test']['pr_auc'] for k in keys], .36, color='#E9C46A', label='Test')
ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_xlabel('ADASYN target minority ratio'); ax[0].set_ylabel('PR-AUC'); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title('(a) Oversampling strength', fontsize=8)
a1 = e['group_share']; a2 = e['adasyn_shap_check']['group_share_alt']; gs = list(a1); x = np.arange(len(gs))
ax[1].bar(x - .18, [a1[g] for g in gs], .36, color='#2A9D8F', label='Deployed (no ADASYN)'); ax[1].bar(x + .18, [a2[g] for g in gs], .36, color='#8D5FD3', label='With ADASYN (0.02)')
ax[1].set_xticks(x); ax[1].set_xticklabels(['History', 'Amount', 'Time', 'Channel/bank/\nlocation/age', 'Anomaly\nscore'], fontsize=7); ax[1].set_ylabel('Share of mean |SHAP|'); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title('(b) Attribution by feature group', fontsize=8)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_adasyn.png', dpi=300); plt.close()
# --- budget sensitivity
fig, ax = plt.subplots(1, 2, figsize=(3.5, 2.3))
Ns = [100, 300, 1000]
for a, (k, t) in zip(ax, (('pr_auc', 'PR-AUC gain'), ('rec_fpr01', 'Recall gain @0.1% FPR'))):
    v = np.array([S[f'N{n}_{k}']['alert-frozen'][3] for n in Ns]); r = np.array([S[f'N{n}_{k}']['alert-random'][3] for n in Ns])
    for vv, c, lb, o in ((v, '#2A9D8F', 'vs frozen', -.04), (r, '#8A8F98', 'vs random', .04)):
        a.errorbar(np.arange(3) + o, vv[:, 0], yerr=[vv[:, 0] - vv[:, 1], vv[:, 2] - vv[:, 0]], fmt='o', color=c, ms=3.5, capsize=2, lw=1, label=lb)
    a.axhline(0, color='k', lw=.6); a.set_xticks(range(3)); a.set_xticklabels(Ns); a.set_xlabel('Reviewed per cycle'); a.set_title(t, fontsize=7.5)
ax[0].legend(frameon=False, fontsize=6.5)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_budget.png', dpi=300); plt.close()
# --- alert card
meta, X, cols, y, tr, va, te = C.load(); D = joblib.load('results/deployed.joblib'); mod = D['model']; names = cols + ['if_anomaly_score']
Xt, yt = X[te], y[te]; p = np.load('results/xai_p_XGB_IF.npy'); mt = meta[te].reset_index(drop=True)
bf = np.flatnonzero((yt == 1) & (p >= D['thr_block'])); i = bf[np.argsort(p[bf])[len(bf) // 2]]
phi = shap.TreeExplainer(mod.model).shap_values(mod.transform(Xt[i:i + 1]))[0]; base = float(np.ravel(shap.TreeExplainer(mod.model).expected_value)[0])
o = np.argsort(-np.abs(phi))[:6][::-1]
lab = {'amount_vs_mean_ratio': 'Amount / customer mean', 'tx_count_total': 'Prior tx count', 'tx_count_24h': 'Tx in last 24 h', 'amount_mean_total': 'Customer mean amount', 'amount_std_total': 'Customer amount std.', 'amount': 'Amount', 'amount_log': 'log amount', 'amount_mean_7d': '7-day mean amount', 'amount_sum_24h': '24 h amount sum', 'amount_std_7d': '7-day amount std.', 'velocity_score': 'Velocity score', 'if_anomaly_score': 'Anomaly score'}
fig, ax = plt.subplots(figsize=(3.5, 2.3))
def fmt(j):
    if j >= len(cols): return 'Anomaly score'
    v = X[te][i][j]; s = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}"
    return lab.get(names[j], names[j].replace('_', ' ')) + ' = ' + s
nm = [fmt(j) for j in o]
ax.barh(nm, phi[o], color=['#E76F51' if v > 0 else '#4C78A8' for v in phi[o]]); ax.axvline(0, color='k', lw=.6)
ax.set_xlabel('SHAP value (log-odds)'); ax.tick_params(axis='y', labelsize=6.5)
ax.set_title(f"BLOCK, score {p[i]:.2f}, amount NGN {mt.amount[i]:,.0f}", fontsize=7.5)
plt.tight_layout(pad=.4); plt.savefig('figures/xai_fig_card.png', dpi=300); print('ok', p[i], mt.channel[i], base)
