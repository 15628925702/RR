"""Small fixed-seed H-estimator comparison; never writes formal JSONL."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from rr_gid_cn.synthetic_oracle import make_frozen_mixture,reference_scale,all_pairs
from rr_gid_cn.s1_gate import run_replication

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--prepared',required=True); ap.add_argument('--out',default='results/p4_h_diagnostic.json'); args=ap.parse_args()
 mix=make_frozen_mixture(seed=2026,alpha=1.0); scale=reference_scale(mix,6000,2026); panels=all_pairs(); p=pickle.load(open(args.prepared,'rb'))
 out=[]
 for seed in (202600000+1000*1000,202600000+1000*1000+1,202600000+1000*1000+2):
  for exact_pilot in (False,True):
   rows=run_replication(mix,scale,panels,1000,seed,prepared=p,lu=8,h_tilted=16,h_cond=2,kl_samples=512,pilot_norm_cap=None,scoring_steps=2,theta_norm_cap=4.0,theta_l1_cap=5.0,kl_mu_direct=False,use_oracle_H=False,diagnostic_exact_pilot=exact_pilot)
   out.extend({'seed':seed,'exact_pilot':exact_pilot,'policy':r['policy'],'B_kl':r['B_kl'],'beta_hat_norm':r['beta_hat_norm'],'diag':r['update_diagnostics']} for r in rows)
 Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
