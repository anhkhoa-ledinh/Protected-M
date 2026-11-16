@echo off
rem Protected M - chay game (giao dien v3 "v2 - Redesign").
rem Bo giao dien cu chay bang:  run_game.bat --ui album
rem Log UTF-8 vao logs\game_<ngay>_<gio>.log (logs\ da gitignore).
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Chua co .venv trong thu muc nay. Tao mot lan bang:
  echo   py -3.13 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG=logs\game_%TS%.log"
echo Protected M - log: %LOG%
set "PYTHONIOENCODING=utf-8"
"%PY%" main.py --demo --album %* > "%LOG%" 2>&1
echo Da thoat (ma %ERRORLEVEL%). Log: %LOG%
