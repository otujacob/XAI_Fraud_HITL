import json, numpy as np
S=json.load(open('results/xai_hitl.json')); rng=np.random.default_rng(0); out={}
def get(N,arm,key,seeds): return np.array([[h[key] for h in S[f'N{N}_s{s}'][arm]] for s in seeds if f'N{N}_s{s}' in S and arm in S[f'N{N}_s{s}']])
def ci(d): 
    b=[rng.choice(d,len(d)).mean() for _ in range(5000)]; return float(d.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
for N,seeds in ((300,range(10)),(100,range(5)),(1000,range(5))):
    for key in ('pr_auc','rec_fpr1','rec_fpr01'):
        r={}
        for arm in ('alert','random','noisy','oracle'):
            if N!=300 and arm in('noisy','oracle'): continue
            a=get(N,arm,key,seeds) if arm!='oracle' else get(N,arm,key,range(5))
            if len(a): r[arm]=[ci(a[:,c]) for c in range(4)]
        a,rd=get(N,'alert',key,seeds),get(N,'random',key,seeds); r['alert-random']=[ci(a[:,c]-rd[:,c]) for c in range(4)]
        f=get(N,'alert',key,seeds)[:,0]; r['alert-frozen']=[ci(a[:,c]-f) for c in range(4)]
        if N==300: n=get(N,'noisy',key,seeds); r['noisy-frozen']=[ci(n[:,c]-f) for c in range(4)]
        out[f'N{N}_{key}']=r
for N in (100,300,1000):
    ff=[np.mean([S[f'N{N}_s{s}'][arm][-1]['true_fraud_found'] for s in range(5)]) for arm in('alert','random')]; print(N,'fraud found alert/random',ff)
json.dump(out,open('results/xai_hitl_summary.json','w'),indent=1)
for k,v in out.items():
    print(k)
    for a,c in v.items(): print('  ',a,[tuple(round(x,4) for x in t) for t in c])
