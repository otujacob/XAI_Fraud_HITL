"""Shared code for the explainable-AI / human-in-the-loop experiments on FraudBench-NG.

Data are split chronologically 60/20/20 (train / validation / test). The validation period doubles as the
analyst-feedback stream in the human-in-the-loop experiment; the test period is never used for fitting,
threshold selection, model selection or feedback."""
import numpy as np, pandas as pd, time
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import ADASYN
import generate_dataset as G

N_LEGIT = 300_000          # legitimate training rows sampled (all training fraud rows are kept)
NJOBS = 2


def load(path='data/nibss_synthetic_fraud.parquet'):
    d = pd.read_parquet(path)
    X = G.model_matrix(d)
    cols = list(X.columns)
    X = X.values.astype(np.float32)
    y = d.is_fraud.values.astype(np.int8)
    ti = d.timestamp.values.astype('int64'); q1, q2 = np.quantile(ti, [.6, .8])
    tr, va, te = ti < q1, (ti >= q1) & (ti < q2), ti >= q2
    meta = d[['customer_id', 'incident_id', 'is_followon', 'timestamp', 'amount', 'channel']].copy()
    return meta, X, cols, y, tr, va, te


def train_idx(y, tr, seed, n_legit=N_LEGIT):
    rng = np.random.default_rng(seed)
    li = np.flatnonzero(tr & (y == 0))
    return np.r_[np.flatnonzero(tr & (y == 1)), rng.choice(li, n_legit, replace=False)]


def rec_at_fpr(yt, p, f, w=None):
    fpr, tpr, _ = roc_curve(yt, p, sample_weight=w)
    return float(np.interp(f, fpr, tpr))


def metrics(yt, p):
    return dict(pr_auc=float(average_precision_score(yt, p)), roc_auc=float(roc_auc_score(yt, p)),
                rec_fpr1=rec_at_fpr(yt, p, .01), rec_fpr01=rec_at_fpr(yt, p, .001),
                prec_top1=float(yt[np.argsort(-p)[:int(.01 * len(p))]].mean()))


def make_xgb(seed, n_estimators=300):
    return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=.8, colsample_bytree=.8,
                             scale_pos_weight=50, n_jobs=NJOBS, random_state=seed, eval_metric='aucpr', tree_method='hist')


def make_rf(seed):
    return RandomForestClassifier(300, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=NJOBS, random_state=seed)


class Detector:
    """scaler -> (optional) Isolation Forest score appended -> (optional) ADASYN on the training sample -> classifier.
    Everything is fitted on the training sample only; transform/predict never refit."""

    def __init__(self, kind='xgb', use_if=False, adasyn=None, seed=0, n_estimators=300):
        self.kind, self.use_if, self.adasyn, self.seed, self.n_est = kind, use_if, adasyn, seed, n_estimators

    def _augment(self, Z):
        return np.c_[Z, -self.iso.score_samples(Z)].astype(np.float32) if self.use_if else Z

    def fit(self, X, y, sample_weight=None):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X).astype(np.float32)
        if self.use_if:
            self.iso = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=NJOBS).fit(Z)
        Z = self._augment(Z)
        w = sample_weight
        if self.adasyn:
            Z, y = ADASYN(sampling_strategy=self.adasyn, n_neighbors=5, random_state=self.seed).fit_resample(Z, y)
            w = None            # feedback weights are not carried through resampling (not used with ADASYN here)
        self.model = (make_xgb(self.seed, self.n_est) if self.kind == 'xgb' else make_rf(self.seed))
        self.model.fit(Z, y, sample_weight=w)
        return self

    def transform(self, X):
        return self._augment(self.scaler.transform(X).astype(np.float32))

    def predict(self, X):
        return self.model.predict_proba(self.transform(X))[:, 1]

    def if_score(self, X):
        return -self.iso.score_samples(self.scaler.transform(X).astype(np.float32))


def clustered_bootstrap(yt, p, cust, B=200, seed=1):
    ncust = cust.max() + 1; rb = np.random.default_rng(seed); out = []
    for _ in range(B):
        w = np.bincount(rb.integers(0, ncust, ncust), minlength=ncust)[cust].astype(float)
        out.append((average_precision_score(yt, p, sample_weight=w), rec_at_fpr(yt, p, .01, w), rec_at_fpr(yt, p, .001, w)))
    out = np.array(out)
    return {k: [float(v) for v in np.percentile(out[:, i], [2.5, 97.5])] for i, k in enumerate(['pr_auc', 'rec_fpr1', 'rec_fpr01'])}
