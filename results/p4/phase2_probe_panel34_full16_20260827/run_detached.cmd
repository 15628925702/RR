@echo off
cd /d G:\0-newResearch\4.RR_GID_CN
set PYTHONPATH=src
set PYTHONUNBUFFERED=1
set RR_GID_CN_CUDA_DEVICE=0
echo STARTED %DATE% %TIME%>> "results\p4\phase2_probe_panel34_full16_20260827\detached.log"
"G:\Anaconda\envs\dermagent-xh\python.exe" scripts\p4_phase2_score_centering.py --config configs\p4_phase2_gold_full16_outer16_20260827.yaml --out results\p4\phase2_probe_panel34_full16_20260827 --resume --only-indices 34
echo EXITCODE=%ERRORLEVEL% %DATE% %TIME%>> "results\p4\phase2_probe_panel34_full16_20260827\detached.log"
