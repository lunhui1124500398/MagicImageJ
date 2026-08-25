@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: MagicImageJ environment is missing.
  echo Run: uv sync
  pause
  exit /b 1
)
"%PY%" "%~dp0main.py"
if errorlevel 1 pause
