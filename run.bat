@echo off
REM Starts the Arabic Historical Characters app on http://localhost:8000
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Creating the Python environment - first run only, this takes a few minutes...
  py -3.10 -m venv venv 2>nul || python -m venv venv
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  venv\Scripts\python.exe -m pip install -r requirements.txt
)
if not exist .env copy .env.example .env >nul
start "" http://localhost:8000
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
