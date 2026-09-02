@echo off
cd /d G:\0-newResearch\4.RR_GID_CN
set PYTHONPATH=src
set PYTHONUNBUFFERED=1
set RR_GID_CN_CUDA_DEVICE=0
echo STARTED %DATE% %TIME%>> "results\p4\_launch_probe_panel34_cond5e5_chunkfix.log"
"G:\Anaconda\envs\dermagent-xh\python.exe" scripts\p4_phase2_score_centering.py --config configs\p4_phase2_gold_full16_cond5e5_20260827.yaml --out results\p4\phase2_probe_panel34_cond5e5_chunkfix_20260828 --only-indices 34
echo EXITCODE=%ERRORLEVEL% %DATE% %TIME%>> "results\p4\_launch_probe_panel34_cond5e5_chunkfix.log"
