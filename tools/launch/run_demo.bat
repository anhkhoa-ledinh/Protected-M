@echo off
rem PB-055: che do demo - chi G, T, B, + = - 0, F10 (relief), F6, keo-tha, F4, Esc co tac dung;
rem doc assets\config\content.demo.json, khong ghi file config nao.
rem Log UTF-8 vao logs\demo_<ngay>_<gio>.log (logs\ da gitignore).
cd /d "%~dp0..\.."
if not exist logs mkdir logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG=logs\demo_%TS%.log"
echo HCPD demo - log: %LOG%
set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" main.py --demo > "%LOG%" 2>&1
echo Da thoat (ma %ERRORLEVEL%). Log: %LOG%
