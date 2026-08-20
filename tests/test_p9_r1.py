from pathlib import Path


def test_p9_config_exists():
    assert Path("configs/p9_smoke.yaml").exists()

