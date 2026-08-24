import argparse,pickle
from pathlib import Path
import numpy as np
from rr_gid_cn.s1_gate import policy_designs
from rr_gid_cn.synthetic_oracle import all_pairs,make_frozen_mixture,reference_scale

ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=Path('experiments/p4_prepared_oracle_hp.pkl')); ap.add_argument('--parts',type=Path,default=Path('experiments/p4_qmc_part')); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
p=pickle.loads(args.source.read_bytes()); infos=[]
for i in range(8):
 d=pickle.loads(Path(f'{args.parts}{i}.pkl').read_bytes()); infos.append(np.asarray(d['information']))
infos=np.concatenate(infos,axis=0); mix=make_frozen_mixture(seed=2026,alpha=1.0); scale=reference_scale(mix,6000,2026)
ref=np.asarray(p['reference']); p['information']=infos; p['designs']=policy_designs(ref,all_pairs(),np.asarray(p['fisher']),infos,fw_tolerance=1e-6); p['_qmc_information']={'shards':8,'tilted':2048,'qmc_order':10}
args.out.write_bytes(pickle.dumps(p,protocol=5)); print({'out':str(args.out),'shape':infos.shape})
