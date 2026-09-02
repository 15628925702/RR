import json
import pickle
from pathlib import Path
from rr_gid_cn.synthetic_oracle import make_frozen_mixture, reference_scale, all_pairs
from rr_gid_cn.s1_gate import run_replication

mix = make_frozen_mixture(seed=2026, alpha=1.0)
scale = reference_scale(mix, 6000, 2026)
panels = all_pairs()
p = pickle.load(open('/root/RR_GID_CN/legacy_server_20260824/experiments/p4_prepared_oracle.pkl', 'rb'))
out = []
for lu in (25, 50, 100):
    rows = run_replication(mix, scale, panels, 100, 202900000,
                           prepared=p, lu=lu, h_tilted=32, h_cond=4,
                           kl_samples=256, pilot_norm_cap=None,
                           scoring_steps=2, theta_norm_cap=4.0,
                           theta_l1_cap=5.0, kl_mu_direct=False,
                           use_oracle_H=False, policies=['oracle RR-GID'])
    out.extend({'lu': lu, 'policy': r['policy'], 'B_kl': r['B_kl'],
                'diag': r['update_diagnostics']} for r in rows)
Path('results/p4_lu_diagnostic.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
