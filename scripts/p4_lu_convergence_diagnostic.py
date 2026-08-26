"""Budget-growing LU convergence probe; output is diagnostic, never formal."""
import json
import pickle
from pathlib import Path

from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale
from rr_gid_cn.s1_gate import run_replication


def main():
    mix = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mix, 6000, 2026)
    panels = all_pairs()
    with open('/root/RR_GID_CN/legacy_server_20260824/experiments/p4_prepared_oracle.pkl', 'rb') as f:
        prepared = pickle.load(f)
    rows = []
    for budget in (100, 200, 400, 800):
        for rep in (0, 1):
            seed = 202600000 + budget * 1000 + rep
            result = run_replication(
                mix, scale, panels, budget, seed, prepared=prepared,
                lu=budget, h_tilted=64, h_cond=8, kl_samples=1024,
                pilot_norm_cap=None, scoring_steps=2,
                theta_norm_cap=4.0, theta_l1_cap=5.0,
                kl_mu_direct=False, use_oracle_H=False,
                policies=['oracle RR-GID'],
            )[0]
            rows.append({
                'budget': budget, 'replication': rep, 'lu': budget,
                'B_kl': result['B_kl'], 'kl': result['kl'],
                'beta_hat_norm': result['beta_hat_norm'],
                'update_diagnostics': result['update_diagnostics'],
            })
            print(json.dumps(rows[-1]), flush=True)
    payload = {
        'stage': 'P4', 'kind': 'budget-growing-LU diagnostic',
        'theory_oracle_half_phi': 63.73693541665105 / 2.0,
        'rows': rows,
    }
    Path('results/p4_lu_convergence_diagnostic.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
