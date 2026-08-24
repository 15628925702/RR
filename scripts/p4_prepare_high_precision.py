"""Build a new high-precision P4 prepared artifact without overwriting legacy data."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
from rr_gid_cn.s1_gate import prepare_s1_oracle
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('experiments/p4_prepared_oracle_hp.pkl'))
    ap.add_argument('--reference-size',type=int,default=200000)
    ap.add_argument('--large-reference-size',type=int,default=1000000)
    ap.add_argument('--information-tilted',type=int,default=1024)
    ap.add_argument('--information-conditional',type=int,default=64)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(f'refusing to overwrite {args.out}')
    mix=make_frozen_mixture(seed=2026,alpha=1.0)
    scale=reference_scale(mix,6000,2026)
    prepared=prepare_s1_oracle(mix,scale,all_pairs(),seed=2026,
        reference_size=args.reference_size,
        information_samples=args.information_tilted,
        conditional_samples=args.information_conditional,
        large_reference_size=args.large_reference_size)
    prepared['_build_parameters']=vars(args)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('wb') as f: pickle.dump(prepared,f,protocol=5)
    print({'out':str(args.out),'reference':len(prepared['reference']),'reference_large':len(prepared['reference_large']), 'information_shape':prepared['information'].shape})
if __name__=='__main__': main()
