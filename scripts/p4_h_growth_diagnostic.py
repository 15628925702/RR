"""P4 diagnostic separating update-information MC from LU.

All rows are written under a diagnostic directory.  Target seeds are paired
across information-sample settings; no formal P4 JSONL is touched.
"""
from __future__ import annotations
import argparse, json, math, pickle
from pathlib import Path
import numpy as np
from rr_gid_cn.s1_gate import run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,default=Path('experiments/p4_prepared_oracle.pkl'))
    ap.add_argument('--out',type=Path,default=Path('results/p4_h_growth_diagnostic.json'))
    ap.add_argument('--budgets',type=int,nargs='+',default=[200,400])
    ap.add_argument('--reps',type=int,default=4)
    ap.add_argument('--h',type=int,nargs=2,action='append',default=[[256,8],[512,16],[1024,32]])
    args=ap.parse_args()
    with args.prepared.open('rb') as f: prepared=pickle.load(f)
    if len(prepared.get('reference',[]))<50000 or len(prepared.get('reference_large',[]))<200000: raise ValueError('small prepared artifact')
    mix=make_frozen_mixture(seed=2026,alpha=1.0); scale=reference_scale(mix,6000,2026); panels=all_pairs(); rows=[]
    for b in args.budgets:
      for rep in range(args.reps):
       seed=202600000+b*1000+rep
       for ht,hc in args.h:
        r=run_replication(mix,scale,panels,b,seed,prepared=prepared,lu=b,h_tilted=ht,h_cond=hc,kl_samples=4096,pilot_norm_cap=None,scoring_steps=2,theta_norm_cap=4.0,theta_l1_cap=5.0,kl_mu_direct=True,mu_direct=True,mu_samples=12000,use_oracle_H=False,policies=['oracle RR-GID'])[0]
        rows.append({'budget':b,'replication':rep,'seed':seed,'h_tilted':ht,'h_cond':hc,'lu':b,'B_kl':r['B_kl'],'B_kl_raw':r.get('B_kl_raw'),'kl':r['kl'],'kl_raw':r.get('kl_raw'),'design_ratio':r['design_ratio'],'update_diagnostics':r['update_diagnostics']}); print(json.dumps(rows[-1],sort_keys=True),flush=True)
    summaries=[]
    for b in args.budgets:
      for ht,hc in args.h:
       x=np.array([r['B_kl_raw'] for r in rows if r['budget']==b and r['h_tilted']==ht and r['h_cond']==hc],float)
       summaries.append({'budget':b,'h_tilted':ht,'h_cond':hc,'n':len(x),'mean_B_kl_raw':float(x.mean()),'sd':float(x.std(ddof=1)) if len(x)>1 else 0.0,'se':float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else 0.0,'ci95_low':float(x.mean()-1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else float(x.mean()),'ci95_high':float(x.mean()+1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else float(x.mean())})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'stage':'P4','kind':'update-information-H growth diagnostic','parameters':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},'target_half_phi':63.73693541665105/2,'rows':rows,'summaries':summaries},indent=2)+'\n')
if __name__=='__main__': main()
