"""Summarize independent P4 probes without touching formal JSONL."""
import argparse, glob, json, math
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument('--pattern',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
rows=[]
for f in sorted(glob.glob(args.pattern)):
    d=json.loads(Path(f).read_text()); rows.append(d)
x=np.asarray([r['B_kl_raw'] for r in rows],float); b=int(rows[0]['budget']); s=float(x.std(ddof=1)) if len(x)>1 else 0.0
payload={'stage':'P4','kind':'diagnostic-only','budget':b,'n':len(x),'rows':rows,
 'mean_B_kl_raw':float(x.mean()),'sd_B_kl_raw':s,'se_B_kl_raw':float(s/math.sqrt(len(x))) if len(x)>1 else 0.0,
 'ci95_low':float(x.mean()-1.96*s/math.sqrt(len(x))) if len(x)>1 else float(x.mean()),
 'ci95_high':float(x.mean()+1.96*s/math.sqrt(len(x))) if len(x)>1 else float(x.mean()),
 'all_raw_nonnegative':bool(np.all(x>=-1e-8)),
 'min_lambda_min_H':float(min(min(z['lambda_min_H'] for z in r['diag'][1:]) for r in rows))}
Path(args.out).write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({k:payload[k] for k in ('budget','n','mean_B_kl_raw','ci95_low','ci95_high','all_raw_nonnegative','min_lambda_min_H')}))
