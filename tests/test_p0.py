import json
import random
from pathlib import Path

import pytest

from rr_gid_cn.config import load_config
from rr_gid_cn.device import select_device
from rr_gid_cn.seed import initialize_seed


def test_config_schema_and_paths(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 7, "device": "cpu", "dtype": "float64", "paths": {"root": ".", "data": "data"}}), encoding="utf-8")
    config = load_config(path)
    assert config.seed == 7
    assert config.paths.data == (tmp_path / "data").resolve()


def test_seed_reproducibility_and_change():
    initialize_seed(11)
    first = [random.random() for _ in range(4)]
    initialize_seed(11)
    assert first == [random.random() for _ in range(4)]
    initialize_seed(12)
    assert first != [random.random() for _ in range(4)]


def test_device_cpu_and_invalid():
    assert select_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        select_device("tpu")


def test_missing_config_key(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing configuration"):
        load_config(path)

