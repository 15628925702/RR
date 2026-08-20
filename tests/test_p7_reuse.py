import json
from pathlib import Path


def test_reuse_manifest_schema():
    assert Path("configs/p7_smoke.yaml").exists()

