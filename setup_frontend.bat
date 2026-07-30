@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist ".venv-frontend\Scripts\python.exe" (
  echo Creating .venv-frontend ...
  python -m venv .venv-frontend
  if errorlevel 1 (
    echo Failed to create venv.
    exit /b 1
  )
)

call ".venv-frontend\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-frontend.txt
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)

if "%GRADIO_SHARE%"=="" set GRADIO_SHARE=False
if "%GRADIO_SERVER_NAME%"=="" set GRADIO_SERVER_NAME=127.0.0.1

echo Starting frontend UI ...
python -m ui.app
