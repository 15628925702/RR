"""Summarize completed P4 paired rows with MC SE and 95% normal CIs."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("paths",nargs="+",type=Path); ap.add_argument("--output",type=Path,default=Path("results/p4_formal_summary.json")); a=ap.parse_args()
    rows=[json.loads(x) for p in a.paths for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    out={}
    for budget in sorted({r["budget"] for r in rows}):
      out[str(budget)]={}
      for policy in sorted({r["policy"] for r in rows}):
        vals=np.asarray([r["B_kl"] for r in rows if r["budget"]==budget and r["policy"]==policy],float)
        out[str(budget)][policy]={"n":int(vals.size),"mean_B_kl":float(vals.mean()),"mc_se":float(vals.std(ddof=1)/np.sqrt(vals.size)),"ci95":[float(vals.mean()-1.96*vals.std(ddof=1)/np.sqrt(vals.size)),float(vals.mean()+1.96*vals.std(ddof=1)/np.sqrt(vals.size))],"mean_design_ratio":float(np.mean([r["design_ratio"] for r in rows if r["budget"]==budget and r["policy"]==policy]))}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
