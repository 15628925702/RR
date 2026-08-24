import argparse, json, pickle, time, traceback
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale
from rr_gid_cn.s1_gate import run_replication

ap = argparse.ArgumentParser(); ap.add_argument('--budget',type=int,required=True); ap.add_argument('--rep',type=int,required=True); ap.add_argument('--h-tilted',type=int,required=True); ap.add_argument('--h-cond',type=int,required=True); ap.add_argument('--qmc-order',type=int,default=8); ap.add_argument('--oracle-h',action='store_true'); ap.add_argument('--prepared',default='experiments/p4_prepared_oracle_hp.pkl'); ap.add_argument('--out',required=True); args=ap.parse_args()
try:
    m=make_frozen_mixture(seed=2026,alpha=1.0); sc=reference_scale(m,6000,2026); p=pickle.load(open(args.prepared,'rb')); t=time.time(); seed=202600000+args.budget*1000+args.rep
    r=run_replication(m,sc,all_pairs(),args.budget,seed,prepared=p,lu=args.budget,h_tilted=args.h_tilted,h_cond=args.h_cond,kl_samples=512,pilot_norm_cap=None,scoring_steps=2,theta_norm_cap=4.0,theta_l1_cap=5.0,kl_mu_direct=False,use_oracle_H=args.oracle_h,policies=['oracle RR-GID'],conditional_method='qmc',qmc_order=args.qmc_order)[0]
    json.dump({'elapsed':time.time()-t,'budget':args.budget,'rep':args.rep,'h_tilted':args.h_tilted,'h_cond':args.h_cond,'qmc_order':args.qmc_order,'B_kl_raw':r['B_kl_raw'],'design_ratio':r['design_ratio'],'diag':r['update_diagnostics']},open(args.out,'w'),indent=2)
except Exception:
    traceback.print_exc(); raise
