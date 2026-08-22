"""Launch the P7 formal (alpha sweep + reuse) as a single GPU process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "scripts" / "p7_formal.py")]
    env = {"PYTHONPATH": str(root / "src")}
    log = Path("/root/p7_logs") / "p7_formal.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(root))
    print(f"P7 launched pid={p.pid} -> {log}", flush=True)


if __name__ == "__main__":
    main()
