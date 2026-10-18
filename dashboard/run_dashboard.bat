@echo off
REM Start the XLS-R deepfake detector dashboard. Extra options are passed through, e.g. run_dashboard.bat --share
setlocal
cd /d "%~dp0"
if exist "D:\anaconda3\Scripts\activate.bat" (
  call "D:\anaconda3\Scripts\activate.bat" xlsr
) else (
  call conda activate xlsr
)
if errorlevel 1 (
  echo Could not activate the xlsr environment. Open Anaconda Prompt and run this script again.
  pause
  exit /b 1
)
if not defined HF_HOME if defined PILOT_DATA_ROOT if exist "%PILOT_DATA_ROOT%\hf_cache" set "HF_HOME=%PILOT_DATA_ROOT%\hf_cache"
python app.py %*
pause
