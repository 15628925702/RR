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
for step in (0.25, 0.5, 0.75, 1.0):
    rows = run_replication(mix, scale, panels, 500, 202800000, prepared=p,
                           lu=8, h_tilted=16, h_cond=2, kl_samples=256,
                           pilot_norm_cap=None, scoring_steps=2,
                           theta_norm_cap=4.0, theta_l1_cap=5.0,
                           kl_mu_direct=False, use_oracle_H=False,
                           scoring_step_size=step)
    out.extend({'step': step, 'policy': r['policy'], 'B_kl': r['B_kl'],
                'diag': r['update_diagnostics']} for r in rows)
Path('results/p4_step_diagnostic.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
