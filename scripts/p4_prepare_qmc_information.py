"""Build an independent P4 artifact with QMC panel information."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
from rr_gid_cn.s1_gate import policy_designs, panel_information_cross
from rr_gid_cn.synthetic_oracle import all_pairs, feature_map, make_frozen_mixture, reference_scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('experiments/p4_prepared_oracle_hp.pkl'))
    ap.add_argument('--out', type=Path, default=Path('experiments/p4_prepared_oracle_qmc.pkl'))
    ap.add_argument('--tilted', type=int, default=4096)
    ap.add_argument('--order', type=int, default=10)
    ap.add_argument('--panel-start', type=int, default=0)
    ap.add_argument('--panel-end', type=int, default=None)
    ap.add_argument('--partial-out', type=Path, default=None)
    args = ap.parse_args()
    if args.out.exists() and args.partial_out is None:
        raise FileExistsError(args.out)
    p = pickle.loads(args.source.read_bytes())
    mix = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mix, 6000, 2026)
    panels = all_pairs()
    panel_end = len(panels) if args.panel_end is None else min(args.panel_end, len(panels))
    panel_subset = panels[args.panel_start:panel_end]
    ref = np.asarray(p['reference'])
    beta = np.asarray(p['beta_true'])
    phi = feature_map(ref, scale)
    logits = phi @ beta
    w = np.exp(logits - logits.max()); w /= w.sum()
    rng = np.random.default_rng(20260017)
    idx = rng.choice(len(ref), size=args.tilted, replace=True, p=w)
    fisher_phi = phi[idx]
    fisher = np.cov(fisher_phi, rowvar=False)
    infos = panel_information_cross(
        mix, beta, panel_subset, ref, scale, args.tilted, 0, 20260019 + args.panel_start,
        conditional_method='qmc', qmc_order=args.order,
    )
    if args.partial_out is not None:
        args.partial_out.write_bytes(pickle.dumps({'start': args.panel_start, 'end': panel_end, 'information': infos}, protocol=5))
        print({'partial': str(args.partial_out), 'start': args.panel_start, 'end': panel_end})
        return
    designs = policy_designs(ref, panels, fisher, infos, fw_tolerance=1e-6)
    out = dict(p)
    out['fisher'] = fisher
    out['information'] = infos
    out['designs'] = designs
    out['_qmc_information'] = {'tilted': args.tilted, 'qmc_order': args.order, 'seed': 20260019}
    args.out.write_bytes(pickle.dumps(out, protocol=5))
    print({'out': str(args.out), 'information': infos.shape, 'tilted': args.tilted, 'order': args.order})

if __name__ == '__main__':
    main()
