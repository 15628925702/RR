"""Summarize completed P4 paired rows with MC SE and 95% normal CIs."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("paths",nargs="+",type=Path); ap.add_argument("--output",type=Path,default=Path("results/p4_formal_summary.json")); ap.add_argument("--integration-tolerance",type=float,required=True); a=ap.parse_args()
    rows=[json.loads(x) for p in a.paths for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    out={}
    for budget in sorted({r["budget"] for r in rows}):
      out[str(budget)]={}
      for policy in sorted({r["policy"] for r in rows}):
        selected=[r for r in rows if r["budget"]==budget and r["policy"]==policy]
        vals=np.asarray([r["B_kl_raw"] for r in selected],float)
        se=float(vals.std(ddof=1)/np.sqrt(vals.size)) if vals.size > 1 else None
        out[str(budget)][policy]={"n":int(vals.size),"mean_B_kl_raw":float(vals.mean()),"mc_se":se,"ci95":[float(vals.mean()-1.96*se),float(vals.mean()+1.96*se)] if se is not None else None,"negative_count":int(np.sum(vals < 0)),"minimum_B_kl_raw":float(vals.min()),"integration_tolerance":float(a.integration_tolerance),"mean_design_ratio_main":float(np.mean([r["design_ratio_main"] for r in selected])),"mean_risk_ratio_raw":float(np.mean([r["risk_ratio_raw"] for r in selected]))}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
