@echo off
REM Launch the Deep Trading System Streamlit app as a server.
REM Usage: run.bat [port]
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=8501

cd /d "%~dp0"
streamlit run app.py --server.address 0.0.0.0 --server.port %PORT%
endlocal
