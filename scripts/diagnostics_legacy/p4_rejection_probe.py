import argparse, json, pickle, time
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale
from rr_gid_cn.s1_gate import run_replication

ap = argparse.ArgumentParser()
ap.add_argument('--budget', type=int, required=True)
ap.add_argument('--rep', type=int, default=0)
ap.add_argument('--prepared', default='experiments/p4_prepared_oracle.pkl')
ap.add_argument('--out', required=True)
ap.add_argument('--lu', type=int, default=64)
ap.add_argument('--h-tilted', type=int, default=16)
ap.add_argument('--h-cond', type=int, default=8)
args = ap.parse_args()
m = make_frozen_mixture(seed=2026, alpha=1.0)
scale = reference_scale(m, 6000, 2026)
with open(args.prepared, 'rb') as f:
    prepared = pickle.load(f)
started = time.time()
rows = run_replication(
    m, scale, all_pairs(), args.budget,
    202600000 + args.budget * 1000 + args.rep,
    prepared=prepared, policies=['oracle RR-GID'],
    lu=args.lu, h_tilted=args.h_tilted, h_cond=args.h_cond,
    kl_samples=128, pilot_norm_cap=None, scoring_steps=2,
    theta_norm_cap=4.0, theta_l1_cap=5.0,
    kl_mu_direct=False, use_oracle_H=False,
    conditional_method='rejection', qmc_order=8,
    qmc_start_order=8, qmc_max_order=10,
    qmc_atol=1e-4, qmc_rtol=1e-3,
)
row = rows[0]
row['elapsed_sec'] = time.time() - started
row['calibration'] = True
row['prepared_reference_size'] = len(prepared.get('reference', []))
row['prepared_reference_large_size'] = len(prepared.get('reference_large', []))
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(row, f, indent=2, sort_keys=True)
print(json.dumps({k: row.get(k) for k in ('budget','replication','policy','elapsed_sec','B_kl_raw','design_ratio','acceptance_rate','ess_fraction','fw_gap','lambda_min')}, sort_keys=True))
