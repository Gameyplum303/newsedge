@echo off
REM Launch the NewsEdge dashboard. Uses a .venv next to this file or one folder up, else Python on PATH.
cd /d "%~dp0"
set PY=python
if exist "..\.venv\Scripts\python.exe" set PY=..\.venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
"%PY%" -m streamlit run app.py
