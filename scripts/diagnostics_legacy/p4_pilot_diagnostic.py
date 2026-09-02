"""Small P4 pilot-consistency diagnostic; no formal results are written."""
import json
import argparse
from pathlib import Path
import pickle
import numpy as np
from rr_gid_cn.synthetic_oracle import make_frozen_mixture, reference_scale, all_pairs, tilted_full_sample, tilted_moments, feature_map
from rr_gid_cn.s1_gate import balanced_pilot_counts, pilot_ht_moment, solve_pilot_beta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepared',default='experiments/p4_prepared_oracle_fix1_debug.pkl'); args=ap.parse_args()
    mix=make_frozen_mixture(seed=2026, alpha=1.0); scale=reference_scale(mix,6000,2026); panels=all_pairs()
    with open(args.prepared,'rb') as f: p=pickle.load(f)
    ref=p['reference']; beta=p['beta_true']; mu_true=tilted_moments(beta,ref,scale)[0]
    exact=solve_pilot_beta(mu_true,ref,scale,norm_cap=4.0,l1_cap=5.0)
    out=[]
    for B in (2000,8000,16000,32000):
        full=tilted_full_sample(mix,beta,B,202600000+B*1000+3,scale)
        b0=int(np.ceil(10*B**(1/3)))
        for b in (b0, min(B,1000), min(B,5000)):
            counts=balanced_pilot_counts(panels,b); obs=[]; cur=0
            for panel,count in zip(panels,counts):
                for row in full[cur:cur+count]: obs.append((panel,row[list(panel)]))
                cur += count
            mu,rho=pilot_ht_moment(obs,counts,panels,scale,ref)
            bh=solve_pilot_beta(mu,ref,scale,norm_cap=4.0,l1_cap=5.0)
            out.append({'B':B,'pilot':int(b),'mu_error':float(np.linalg.norm(mu-mu_true)),'beta_error':float(np.linalg.norm(bh-beta)),'beta_norm':float(np.linalg.norm(bh)),'rho_min':float(rho[rho>0].min())})
    result={'beta_true_norm':float(np.linalg.norm(beta)),'exact_mu_beta_error':float(np.linalg.norm(exact-beta)),'exact_mu_solution_norm':float(np.linalg.norm(exact)),'rows':out}
    Path('results').mkdir(exist_ok=True); Path('results/p4_pilot_diagnostic_fix.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
