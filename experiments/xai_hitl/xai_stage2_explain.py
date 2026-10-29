"""Stage 2: SHAP explanations of the deployed detector on the test period - additivity, deletion faithfulness,
stability, what the explanations reveal (fraud vs false alerts), amount counterfactuals, and per-transaction latency.
Writes results/xai_explain.json (+ example alerts and arrays for figures)."""
import os, json, time, platform, joblib, numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
import xai_common as C
import generate_dataset as G

meta, X, cols, y, tr, va, te = C.load()
D = joblib.load('results/deployed.joblib'); m = D['model']; thr_b, thr_r = D['thr_block'], D['thr_review']
names = cols + (['if_anomaly_score'] if m.use_if else [])
Xt, yt = X[te], y[te]; Xtr = X[C.train_idx(y, tr, 0)]
pt = m.predict(Xt)
alerts = np.flatnonzero(pt >= thr_r)               # review + block tiers
print('alerts', len(alerts), 'fraud among alerts', int(yt[alerts].sum()), flush=True)

def group(n):
    if n in ('amount', 'amount_log'): return 'Amount'
    if n.startswith(('channel_', 'bank_', 'location_', 'age_group_')): return 'Channel, bank, location, age'
    if n.startswith(('hour_', 'day_', 'month_')) or n == 'is_peak_hour': return 'Time'
    if n == 'if_anomaly_score': return 'Anomaly score'
    return 'Customer history'
GROUPS = ['Customer history', 'Amount', 'Time', 'Channel, bank, location, age', 'Anomaly score']
gidx = np.array([GROUPS.index(group(n)) for n in names])

booster = m.model; expl = shap.TreeExplainer(booster)
Zt = m.transform(Xt)
rng = np.random.default_rng(0)
legit_sample = rng.choice(np.flatnonzero(yt == 0), 4000, replace=False)
sample = np.unique(np.r_[np.flatnonzero(yt == 1), legit_sample, alerts])
t0 = time.time(); phi = expl.shap_values(Zt[sample]); base = float(np.ravel(expl.expected_value)[0])
print('shap', phi.shape, f'{time.time()-t0:.0f}s', flush=True)
pos = {s: i for i, s in enumerate(sample)}
out = dict(n_alerts=int(len(alerts)), n_alert_fraud=int(yt[alerts].sum()), deployed=D['name'], features=names)

# ---- global attribution by group and top features (all fraud rows + 4,000 legitimate rows, as in the dataset paper)
gs = np.r_[np.flatnonzero(yt == 1), legit_sample]; A = np.abs(phi[[pos[s] for s in gs]]).mean(0)
out['group_share'] = {g: float(A[gidx == i].sum() / A.sum()) for i, g in enumerate(GROUPS)}
top = np.argsort(-A)[:10]; out['top_features'] = {names[i]: float(A[i]) for i in top}


# ---- explainability-imbalance check (Zafar & Wu 2026): do attributions change when the model is trained without/with ADASYN?
idx0 = C.train_idx(y, tr, 0); alt = C.Detector('xgb', m.use_if, None if m.adasyn else D['adasyn'], 0).fit(X[idx0], y[idx0])
Za2 = alt.transform(Xt[gs]); ph2 = shap.TreeExplainer(alt.model).shap_values(Za2); A2 = np.abs(ph2).mean(0)
from scipy.stats import spearmanr as _sp
rho_g = _sp(A, A2)[0]; t5a, t5b = set(np.argsort(-A)[:5]), set(np.argsort(-A2)[:5])
out['adasyn_shap_check'] = dict(deployed_has_adasyn=bool(m.adasyn), spearman_global=float(rho_g), top5_overlap=len(t5a & t5b),
    group_share_alt={g: float(A2[gidx == i].sum() / A2.sum()) for i, g in enumerate(GROUPS)}, alt_test_pr_auc=float(C.metrics(yt, alt.predict(Xt))['pr_auc']),
    top_alt=[names[i] for i in np.argsort(-A2)[:5]], top_dep=[names[i] for i in np.argsort(-A)[:5]])
print('adasyn shap check', out['adasyn_shap_check'], flush=True)

# ---- completeness: base value + sum of attributions equals the model's log-odds output
margin = booster.predict(Zt[sample], output_margin=True) if False else m.model.get_booster().predict(__import__('xgboost').DMatrix(Zt[sample]), output_margin=True)
add_err = np.abs(base + phi.sum(1) - margin); out['additivity_max_abs_error'] = float(add_err.max())

# ---- deletion faithfulness on alerts: replace the top-k attributed features by their training median vs k random features
Za = Zt[alerts]; pa = pt[alerts]; Pa = phi[[pos[s] for s in alerts]]
med = np.median(m.transform(Xtr), axis=0).astype(np.float32)
order = np.argsort(-np.abs(Pa), axis=1)
def score_after(Zm): return booster.predict_proba(Zm)[:, 1]
fa = {}
for k in (1, 3, 5, 10):
    Zs = Za.copy(); r = np.arange(len(Za))[:, None]; Zs[r, order[:, :k]] = med[order[:, :k]]
    drop_shap = pa - score_after(Zs); cleared_shap = float((score_after(Zs) < thr_r).mean())
    dr, cr = [], []
    for rep in range(5):
        rr = np.argsort(rng.random(Za.shape), axis=1)[:, :k]
        Zr = Za.copy(); Zr[np.arange(len(Za))[:, None], rr] = med[rr]
        dr.append((pa - score_after(Zr)).mean()); cr.append(float((score_after(Zr) < thr_r).mean()))
    fa[k] = dict(mean_drop_shap=float(drop_shap.mean()), mean_drop_random=float(np.mean(dr)),
                 cleared_shap=cleared_shap, cleared_random=float(np.mean(cr)), mean_score=float(pa.mean()))
out['faithfulness'] = fa; print('faithfulness', fa, flush=True)

# ---- stability: re-explain after adding small Gaussian noise (0.05 SD) to the continuous standardised features
cont = np.array([not (n.startswith(('channel_', 'bank_', 'location_', 'age_group_')) or n == 'is_peak_hour') for n in names])
sub = rng.choice(len(alerts), 500, replace=False); Zn = Za[sub].copy()
Zn[:, cont] += rng.normal(0, 0.05, (len(sub), int(cont.sum()))).astype(np.float32)
phin = expl.shap_values(Zn); Po = Pa[sub]
rho = [spearmanr(np.abs(Po[i]), np.abs(phin[i]))[0] for i in range(len(sub))]
jac = [len(set(np.argsort(-np.abs(Po[i]))[:5]) & set(np.argsort(-np.abs(phin[i]))[:5])) / len(set(np.argsort(-np.abs(Po[i]))[:5]) | set(np.argsort(-np.abs(phin[i]))[:5])) for i in range(len(sub))]
same1 = float(np.mean([np.argmax(np.abs(Po[i])) == np.argmax(np.abs(phin[i])) for i in range(len(sub))]))
out['stability'] = dict(spearman_mean=float(np.nanmean(rho)), spearman_p10=float(np.nanpercentile(rho, 10)), top5_jaccard_mean=float(np.mean(jac)),
                        top1_same=same1, mean_abs_score_change=float(np.abs(score_after(Zn) - pa[sub]).mean()))
print('stability', out['stability'], flush=True)

# ---- what do explanations reveal? top-1 feature group and most frequent top-3 features, true fraud vs false alerts
isf = yt[alerts] == 1
top1 = order[:, 0]
out['top1_group'] = {lab: {g: float((gidx[top1[msk]] == i).mean()) for i, g in enumerate(GROUPS)} for lab, msk in (('fraud', isf), ('false_alert', ~isf))}
def freq(msk):
    c = pd.Series(np.array(names)[order[msk][:, :3].ravel()]).value_counts(normalize=True) * 3
    return {k: float(v) for k, v in c.head(6).items()}
out['top3_freq'] = dict(fraud=freq(isf), false_alert=freq(~isf))
signed = {lab: {names[i]: float(Pa[msk][:, i].mean()) for i in np.argsort(-np.abs(Pa[msk]).mean(0))[:6]} for lab, msk in (('fraud', isf), ('false_alert', ~isf))}
out['mean_signed_shap'] = signed

# ---- counterfactual: smallest reduction of the transaction amount that moves an alert below the REVIEW threshold.
# amount, amount_log and amount_vs_mean_ratio are recomputed consistently; all other features depend on prior history only.
ia, il, ir = cols.index('amount'), cols.index('amount_log'), cols.index('amount_vs_mean_ratio')
scales = np.r_[1, np.geomspace(0.95, 0.01, 40)]
Xa = Xt[alerts]; clear = np.full(len(alerts), np.nan)
for s in range(0, len(alerts), 500):
    xb = Xa[s:s + 500]; nb = len(xb)
    big = np.repeat(xb, len(scales), axis=0); sc = np.tile(scales, nb)
    big[:, ia] = np.maximum(xb[:, ia].repeat(len(scales)) * sc, 1); big[:, il] = np.log1p(big[:, ia]); big[:, ir] = xb[:, ir].repeat(len(scales)) * sc
    ps = m.predict(big).reshape(nb, len(scales))
    for i in range(nb):
        ok = np.flatnonzero(ps[i] < thr_r)
        if len(ok): clear[s + i] = 1 - scales[ok[0]]            # first (smallest) reduction that clears the alert
cl = ~np.isnan(clear)
out['counterfactual'] = dict(share_clearable=float(cl.mean()), share_clearable_fraud=float(cl[isf].mean()), share_clearable_false=float(cl[~isf].mean()),
                             share_clear_le50=float((clear <= .5).mean()) if True else None,
                             median_reduction=float(np.nanmedian(clear)), median_reduction_fraud=float(np.nanmedian(clear[isf])) if cl[isf].any() else None,
                             median_reduction_false=float(np.nanmedian(clear[~isf])) if cl[~isf].any() else None, n=int(len(alerts)))
print('counterfactual', out['counterfactual'], flush=True)

# ---- explanation text for example alerts
def fmt(n, v):
    if n in ('amount', 'amount_mean_7d', 'amount_std_7d', 'amount_sum_24h', 'amount_mean_total', 'amount_std_total'): return f'NGN {v:,.0f}'
    if n.endswith('ratio') or n in ('velocity_score',): return f'{v:.2f}'
    return f'{v:.2f}' if abs(v) < 100 and v != int(v) else f'{v:,.0f}'
def explain_text(i):                      # i indexes the alerts array
    j = pos[alerts[i]]; row = Xt[alerts[i]]; ph = phi[j]; o = np.argsort(-np.abs(ph))[:3]; parts = []
    for f in o:
        v = row[f] if f < len(cols) else float(m.if_score(row[None])[0])
        parts.append(f'{G.label(names[f]) if names[f] in cols else "Isolation Forest anomaly score"} = {fmt(names[f], v)} ({"raises" if ph[f] > 0 else "lowers"} risk, {ph[f]:+.2f})')
    return '; '.join(parts)
ex = []
tp_i = [i for i in np.argsort(-pa) if isf[i]][:1] + [i for i in np.argsort(-pa) if not isf[i]][:1]
for i in tp_i:
    ex.append(dict(fraud=bool(isf[i]), score=float(pa[i]), tier='BLOCK' if pa[i] >= thr_b else 'REVIEW', amount=float(Xt[alerts[i], ia]), text=explain_text(i),
                   cf_reduction=None if np.isnan(clear[i]) else float(clear[i])))
out['examples'] = ex; print(ex, flush=True)

# ---- latency: feature vector -> IF score -> classifier -> SHAP -> text -> counterfactual, one transaction at a time, one thread
m.model.set_params(n_jobs=1); m.iso.n_jobs = 1
lat_idx = rng.choice(len(Xt), 1000, replace=False); comp = dict(prep=[], score=[], shap=[], text=[], cf=[], total=[])
sc_ = np.r_[1, np.geomspace(0.95, 0.01, 40)]
_ = m.predict(Xt[lat_idx[:5]]); _ = expl.shap_values(Zt[lat_idx[:2]])        # warm-up
for i in lat_idx:
    row = Xt[i:i + 1]; a = time.perf_counter()
    z = m.transform(row); b = time.perf_counter()
    p = booster.predict_proba(z)[:, 1]; c = time.perf_counter()
    ph = expl.shap_values(z)[0]; d_ = time.perf_counter()
    o = np.argsort(-np.abs(ph))[:3]; txt = '; '.join(f'{names[f]} {ph[f]:+.2f}' for f in o); e = time.perf_counter()
    big = np.repeat(row, len(sc_), axis=0); big[:, ia] = np.maximum(row[0, ia] * sc_, 1); big[:, il] = np.log1p(big[:, ia]); big[:, ir] = row[0, ir] * sc_
    _ = m.predict(big); f_ = time.perf_counter()
    comp['prep'].append(b - a); comp['score'].append(c - b); comp['shap'].append(d_ - c); comp['text'].append(e - d_); comp['cf'].append(f_ - e); comp['total'].append(f_ - a)
ms = lambda v: dict(median=float(np.median(v) * 1e3), p95=float(np.percentile(v, 95) * 1e3), p99=float(np.percentile(v, 99) * 1e3), mean=float(np.mean(v) * 1e3))
out['latency_ms'] = {k: ms(v) for k, v in comp.items()}
cpu = [l.split(':')[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][:1]
out['latency_hw'] = dict(cpu=cpu[0] if cpu else platform.processor(), threads=1, n=1000)
print('latency', out['latency_ms'], flush=True)
np.savez('results/xai_explain_arrays.npz', alerts=alerts, yt_alert=yt[alerts], pa=pa, clear=clear, group_idx=gidx,
         shap_mean_abs=A, names=np.array(names))
json.dump(out, open('results/xai_explain.json', 'w'), indent=1)
print('done')
