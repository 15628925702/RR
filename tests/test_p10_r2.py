from pathlib import Path


def test_p10_config_exists():
    assert Path("configs/p10_smoke.yaml").exists()

