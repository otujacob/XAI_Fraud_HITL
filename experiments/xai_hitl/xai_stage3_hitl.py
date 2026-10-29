"""Stage 3: human-in-the-loop feedback simulation on FraudBench-NG, properly powered.

Timeline: the detector is trained on the training period (60%). The validation period (next 20%) is the analyst-feedback
stream, split into three chronological cycles. In each cycle the CURRENT model scores that cycle's transactions and the
analysts review the N highest-scored ones; their decisions are added to the training data and the classifier is retrained
(scaler and Isolation Forest stay fixed). After every cycle the model is evaluated on the fixed test period (last 20%), which is
never used for feedback, thresholds or model selection.

Arms: frozen (no feedback); alert (analysts review the top-N alerts, perfect labels); random (N random transactions are
labelled, same cost - separates 'more labels' from 'labels chosen by alerts'); noisy (alert review, analysts miss 10% of frauds
and wrongly confirm 2% of legitimate alerts); oracle (every fraud of the cycle plus 20,000 random legitimate rows is labelled).
Each seed redraws the training sample and the classifier seed, so seed-to-seed spread includes training variability and
differences are paired within a seed. Writes results/xai_hitl.json (incremental, resumable)."""
import os, json, joblib, sys, time, numpy as np
from imblearn.over_sampling import ADASYN
import xai_common as C

D = joblib.load('results/deployed.joblib'); name = D['name']; R = D['adasyn'] if 'ADASYN' in name else None
USE_IF = 'IF' in name
MAIN_N = 300; MAIN_SEEDS = list(range(10)); EXTRA_N = [100, 1000]; EXTRA_SEEDS = list(range(5)); NOISE = dict(miss=0.10, false_confirm=0.02)
meta, X, cols, y, tr, va, te = C.load()
Xt, yt = X[te], y[te]
ti = meta.timestamp.values.astype('int64'); vi = np.flatnonzero(va)
edges = np.quantile(ti[vi], [0, 1 / 3, 2 / 3, 1]); blocks = [vi[(ti[vi] >= edges[k]) & (ti[vi] <= edges[k + 1] if k == 2 else ti[vi] < edges[k + 1])] for k in range(3)]
print('cycle sizes / fraud', [(len(b), int(y[b].sum())) for b in blocks], 'deployed', name, 'adasyn', R, flush=True)
PATH = 'results/xai_hitl.json'
store = json.load(open(PATH)) if os.path.exists(PATH) else {}

def retrain(m0, Zb, yb, Xf, yf, seed):
    Z = np.r_[Zb, m0.transform(Xf)] if len(Xf) else Zb; yy = np.r_[yb, yf] if len(Xf) else yb
    if R: Z, yy = ADASYN(sampling_strategy=R, n_neighbors=5, random_state=seed).fit_resample(Z, yy)
    mdl = C.make_xgb(seed).fit(Z, yy)
    return lambda A: mdl.predict_proba(m0.transform(A))[:, 1]

def evaluate(f):
    p = f(Xt); mt = C.metrics(yt, p); return {k: mt[k] for k in ('pr_auc', 'rec_fpr1', 'rec_fpr01')}

def run(seed, N, arms):
    key = f'N{N}_s{seed}'
    if key in store and all(a in store[key] for a in arms): return
    t0 = time.time(); rng = np.random.default_rng(1000 + seed)
    idx = C.train_idx(y, tr, seed); Xs, ys = X[idx], y[idx]
    m0 = C.Detector('xgb', USE_IF, R, seed).fit(Xs, ys); Zb = m0.transform(Xs)
    res = store.get(key, {}); res['frozen'] = [evaluate(m0.predict)] * 4
    f0 = m0.predict
    for arm in arms:
        if arm == 'frozen': continue
        f = f0; fb_X, fb_y, fb_idx = np.zeros((0, X.shape[1]), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int64); hist = [evaluate(f0) | dict(true_fraud_found=0, labelled=0)]
        for c in range(3):
            b = blocks[c]
            if arm in ('alert', 'noisy'):
                pick = b[np.argsort(-f(X[b]))[:N]]; lab = y[pick].astype(np.int8)
                if arm == 'noisy':
                    flip = np.where(lab == 1, rng.random(len(lab)) < NOISE['miss'], rng.random(len(lab)) < NOISE['false_confirm']); lab = np.where(flip, 1 - lab, lab).astype(np.int8)
            elif arm == 'random':
                pick = rng.choice(b, N, replace=False); lab = y[pick]
            elif arm == 'oracle':
                pick = np.r_[b[y[b] == 1], rng.choice(b[y[b] == 0], 20000, replace=False)]; lab = y[pick]
            fb_X, fb_y, fb_idx = np.r_[fb_X, X[pick]], np.r_[fb_y, lab], np.r_[fb_idx, pick]
            f = retrain(m0, Zb, ys, fb_X, fb_y, seed)
            hist.append(evaluate(f) | dict(true_fraud_found=int(y[fb_idx].sum()), labelled=int(len(fb_idx))))
        res[arm] = hist
    store[key] = res; json.dump(store, open(PATH, 'w'))
    print(key, arms, {a: [round(h['rec_fpr1'], 3) for h in res[a]] for a in res if a != 'frozen'}, f'{time.time()-t0:.0f}s', flush=True)

for s in MAIN_SEEDS: run(s, MAIN_N, ['alert', 'random', 'noisy'])
for s in EXTRA_SEEDS: run(s, MAIN_N, ['oracle'])
for N in EXTRA_N:
    for s in EXTRA_SEEDS: run(s, N, ['alert', 'random'])
print('done')
