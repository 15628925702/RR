param([string]$Config = "configs/p0_smoke.yaml")
$env:PYTHONPATH = "src"
python -m rr_gid_cn --config $Config --write-manifest

