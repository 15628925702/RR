"""P12: build the frozen-results manifest (sha256 prefix + size per result file)."""
from __future__ import annotations

import hashlib
import json
import os
import glob


def main() -> None:
    manifest = {}
    files = sorted(glob.glob("results/*.jsonl") + glob.glob("results/*.json"))
    for fp in files:
        if "p4_exact" in fp or "p4_diagnostic" in fp or "p4_s1_gate" in fp or "p4_summary_stdout" in fp:
            continue  # large / intermediate diagnostics
        h = hashlib.sha256(open(fp, "rb").read()).hexdigest()[:16]
        manifest[fp.replace("\\", "/")] = {"sha256_prefix": h, "bytes": os.path.getsize(fp)}
    out = {
        "project": "RR-GID_CN",
        "freeze_commit": "f36c802",
        "generated_by": "P12 freeze",
        "files": manifest,
    }
    with open("results/manifest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest files: {len(manifest)}")
    for k in sorted(manifest)[:5]:
        print(" ", k, manifest[k])


if __name__ == "__main__":
    main()
