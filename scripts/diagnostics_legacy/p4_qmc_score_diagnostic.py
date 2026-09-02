"""Nested QMC conditional-score convergence check; never formal output."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from rr_gid_cn.synthetic_oracle import (all_pairs, feature_map, make_frozen_mixture,
    reference_scale, sample_full, tilted_conditional_mean_qmc)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('results/p4_qmc_score_diagnostic.json')); ap.add_argument('--orders',type=int,nargs='+',default=[8,10,12]); ap.add_argument('--rows',type=int,default=8); args=ap.parse_args()
    mix=make_frozen_mixture(seed=2026,alpha=1.0); scale=reference_scale(mix,6000,2026); panels=all_pairs(); full=sample_full(mix,args.rows,202607,)
    beta=np.asarray([0.11,-0.09,0.07,-0.05,0.03,-0.02,0.08,-0.06,0.04,-0.03,0.02,-0.01])
    rows=[]
    for panel in panels[:4]:
      x=full[:,list(panel)]
      for order in args.orders:
       m=tilted_conditional_mean_qmc(mix,beta,x,panel,order,seed=202600+order,scale=scale)
       rows.append({'panel':panel,'order':order,'mean_norm':float(np.linalg.norm(m.mean(0))),'mean':m.mean(0).tolist()})
    errors=[]
    for panel in panels[:4]:
      ref=np.asarray([r['mean'] for r in rows if r['panel']==panel and r['order']==max(args.orders)])
      for order in args.orders[:-1]:
       cur=np.asarray([r['mean'] for r in rows if r['panel']==panel and r['order']==order]); errors.append({'panel':panel,'order':order,'vs_max_order':max(args.orders),'max_abs_error':float(np.max(np.abs(cur-ref)))})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'stage':'P4','kind':'nested QMC conditional score diagnostic','rows':rows,'errors':errors},indent=2)+'\n'); print(json.dumps(errors,indent=2))
if __name__=='__main__': main()
