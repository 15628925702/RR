"""Launch the P10 R2 formal (Gas natural drift) for one campaign."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--campaign", default=None)
    args, _ = ap.parse_known_args()
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "scripts" / "p10_r2_formal.py")]
    if args.campaign:
        cmd += ["--campaign", args.campaign]
    env = {"PYTHONPATH": str(root / "src")}
    log = Path("/root/p10_logs") / f"p10_{args.campaign or 'all'}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(root))
    print(f"P10({args.campaign or 'all'}) launched pid={p.pid} -> {log}", flush=True)


if __name__ == "__main__":
    main()
