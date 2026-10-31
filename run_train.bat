@echo off
if exist .venv\Scripts\python.exe (set "PYTHON=.venv\Scripts\python.exe") else (set "PYTHON=python")
%PYTHON% train.py --config configs/pilot.yaml %*
exit /b %errorlevel%
