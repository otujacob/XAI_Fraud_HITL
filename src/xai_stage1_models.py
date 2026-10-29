"""Stage 1: detector configurations on FraudBench-NG (chronological 60/20/20 split).
Configurations: RF; IF alone; RF+ADASYN; XGB; XGB+IF; XGB+IF+ADASYN; hybrid ensemble (IF + ADASYN, mean of XGB and RF).
The ADASYN strength and the deployed XGBoost variant are chosen on the VALIDATION period; the test period is only
used for reporting. Writes results/xai_models.json, results/deployed.joblib, results/xai_p_*.npy"""
import os; os.makedirs('results', exist_ok=True)
import json, joblib, numpy as np, time
from sklearn.metrics import average_precision_score, precision_recall_curve
import xai_common as C

SEED = 0
meta, X, cols, y, tr, va, te = C.load()
print('rows', len(y), 'features', X.shape[1], '| fraud tr/va/te', y[tr].sum(), y[va].sum(), y[te].sum(), flush=True)
idx = C.train_idx(y, tr, SEED); Xs, ys = X[idx], y[idx]
Xv, yv, Xt, yt = X[va], y[va], X[te], y[te]
cust = np.unique(meta.customer_id.values[te], return_inverse=True)[1]
out = dict(n_train_sample=int(len(idx)), n_train_fraud=int(ys.sum()), n_val=int(va.sum()), n_val_fraud=int(yv.sum()),
           n_test=int(te.sum()), n_test_fraud=int(yt.sum()), features=cols)
t0 = time.time()

# ---- ADASYN strength: chosen on validation PR-AUC (XGB + IF + ADASYN)
abl = {}
for r in [None, 0.01, 0.02, 0.05, 0.20]:
    m = C.Detector('xgb', True, r, SEED).fit(Xs, ys)
    pv, pt = m.predict(Xv), m.predict(Xt)
    abl[str(r)] = dict(val_pr_auc=float(average_precision_score(yv, pv)), test=C.metrics(yt, pt), val=C.metrics(yv, pv))
    print('adasyn', r, round(abl[str(r)]['val_pr_auc'], 4), round(abl[str(r)]['test']['pr_auc'], 4), f'{time.time()-t0:.0f}s', flush=True)
out['adasyn_ablation'] = abl
best_r = max([r for r in abl if r != 'None'], key=lambda r: abl[r]['val_pr_auc']); R = float(best_r)
out['adasyn_selected'] = R; print('selected ADASYN ratio', R, flush=True)

# ---- configurations
cfg = {}; P = {}; models = {}
def evaluate(name, pv, pt):
    cfg[name] = dict(val=C.metrics(yv, pv), test=C.metrics(yt, pt), val_pr_auc=float(average_precision_score(yv, pv)))
    cfg[name]['ci'] = C.clustered_bootstrap(yt, pt, cust)
    P[name] = pt
    print(name, {k: round(v, 4) for k, v in cfg[name]['test'].items()}, f'{time.time()-t0:.0f}s', flush=True)

rf = C.Detector('rf', False, None, SEED).fit(Xs, ys); evaluate('RF', rf.predict(Xv), rf.predict(Xt))
m_if = C.Detector('xgb', True, None, SEED).fit(Xs, ys)
sv, st = m_if.if_score(Xv), m_if.if_score(Xt); evaluate('IF alone', sv, st)
rfa = C.Detector('rf', False, R, SEED).fit(Xs, ys); evaluate('RF+ADASYN', rfa.predict(Xv), rfa.predict(Xt))
xg = C.Detector('xgb', False, None, SEED).fit(Xs, ys); evaluate('XGB', xg.predict(Xv), xg.predict(Xt)); models['XGB'] = xg
evaluate('XGB+IF', m_if.predict(Xv), m_if.predict(Xt)); models['XGB+IF'] = m_if
xga = C.Detector('xgb', True, R, SEED).fit(Xs, ys); evaluate('XGB+IF+ADASYN', xga.predict(Xv), xga.predict(Xt)); models['XGB+IF+ADASYN'] = xga
rfia = C.Detector('rf', True, R, SEED).fit(Xs, ys)
evaluate('Hybrid ensemble', (xga.predict(Xv) + rfia.predict(Xv)) / 2, (xga.predict(Xt) + rfia.predict(Xt)) / 2)
out['configs'] = cfg

# deployed model = best XGBoost variant on validation PR-AUC (TreeExplainer needs a single tree model)
dep = max(models, key=lambda k: cfg[k]['val_pr_auc']); out['deployed'] = dep
m = models[dep]; pv = m.predict(Xv)
# three-tier operating points fixed on validation legitimate rows: BLOCK = 0.1% FPR, REVIEW = 1% FPR
leg = np.sort(pv[yv == 0]); thr_block = float(np.quantile(leg, 1 - .001)); thr_review = float(np.quantile(leg, 1 - .01))
pt = P[dep]; tiers = {}
for nm, mask in (('block', pt >= thr_block), ('review', (pt >= thr_review) & (pt < thr_block)), ('approve', pt < thr_review)):
    tiers[nm] = dict(n=int(mask.sum()), fraud=int(yt[mask].sum()), legit=int((yt[mask] == 0).sum()))
out['thresholds'] = dict(block=thr_block, review=thr_review); out['tiers_test'] = tiers
print('deployed', dep, thr_block, thr_review, tiers, flush=True)
json.dump(out, open('results/xai_models.json', 'w'), indent=1)
joblib.dump(dict(model=m, name=dep, thr_block=thr_block, thr_review=thr_review, cols=cols, adasyn=R), 'results/deployed.joblib')
for k, v in P.items(): np.save('results/xai_p_' + k.replace('+', '_').replace(' ', '_') + '.npy', v.astype(np.float32))
np.save('results/xai_te_idx.npy', np.flatnonzero(te)); np.save('results/xai_yt.npy', yt)
print('done', f'{time.time()-t0:.0f}s')
