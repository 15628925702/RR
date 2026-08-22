"""Launch the P9 R1 formal (Gas budget curves) as a single GPU process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "scripts" / "p9_r1_formal.py")]
    env = {"PYTHONPATH": str(root / "src")}
    log = Path("/root/p9_logs") / "p9_r1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(root))
    print(f"P9 launched pid={p.pid} -> {log}", flush=True)


if __name__ == "__main__":
    main()
