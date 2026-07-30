@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv-server\Scripts\python.exe" (
  echo Creating .venv-server ...
  python -m venv .venv-server
  if errorlevel 1 (
    echo Failed to create venv.
    exit /b 1
  )
)

call ".venv-server\Scripts\activate.bat"
REM requirements-server.txt は追加パッケージなし（コメントのみ）

if "%LLAMACPP_CHAT2_CONTROL_TOKEN%"=="" set LLAMACPP_CHAT2_CONTROL_TOKEN=llamacpp-chat2

echo Starting control API ...
python -m server.control_api
