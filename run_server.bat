@echo off
rem Protected M - chay server PvP LAN (TCP 47821 + UDP discovery 47822).
rem Khong mo webcam, khong mo cua so GL - chi la mot tien trinh server, chay
rem truoc tren mot may roi de nguyen cua so nay mo (Ctrl+C de dung).
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Chua co .venv trong thu muc nay. Tao mot lan bang:
  echo   py -3.13 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Protected M - PvP server. Nho cho Python qua Windows Firewall khi duoc
echo hoi (xem README.md). Ctrl+C de dung server.
set "PYTHONIOENCODING=utf-8"
"%PY%" -m src.net.server -v %*
echo Da thoat (ma %ERRORLEVEL%).
