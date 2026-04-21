@echo off
setlocal EnableDelayedExpansion

set MIN_PYTHON=3.12
set VENV_DIR=%~dp0.venv
set SCRIPT_DIR=%~dp0

echo ========================================
echo   ASR-CLI Windows Installation Script
echo ========================================
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python %MIN_PYTHON% or higher.
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

for /f "delims=" %%i in ('python -c "import sys; print(sys.version_info.major*100 + sys.version_info.minor)"') do set PYVER=%%i
if %PYVER% LSS %MIN_PYTHON% (
    echo ERROR: Python %MIN_PYTHON% or higher required.
    echo You have Python !PYVER!, but this script needs Python %MIN_PYTHON%+.
    echo.
    echo Please upgrade Python:
    echo   https://www.python.org/downloads/
    exit /b 1
)
echo   - Python version OK

echo.
echo [2/5] Setting up virtual environment...
if not exist "%VENV_DIR%" (
    echo   - Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo   - Reusing existing virtual environment...
)
echo   - Virtual environment ready

echo.
echo [3/5] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    exit /b 1
)
echo   - pip upgraded

echo.
echo [4/5] Installing dependencies...
echo   - Installing faster-whisper...
"%VENV_DIR%\Scripts\pip.exe" install faster-whisper
if errorlevel 1 (
    echo ERROR: Failed to install faster-whisper.
    exit /b 1
)
echo   - faster-whisper installed

echo   - Installing av (for audio/video decoding)...
"%VENV_DIR%\Scripts\pip.exe" install av
if errorlevel 1 (
    echo WARNING: av installation failed. Some formats may not work.
)
echo   - av installed

echo.
echo [5/5] Installing asr-cli...
"%VENV_DIR%\Scripts\pip.exe" install -e "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to install asr-cli.
    exit /b 1
)
echo   - asr-cli installed

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Usage:
echo   %VENV_DIR%\Scripts\asr-cli.exe transcribe "audio.mp3" --format srt --model tiny
echo.
echo Or add to PATH:
echo   set PATH=%%PATH%%;%VENV_DIR%\Scripts
echo.
echo For model download, run once with --model small (requires internet):
echo   %VENV_DIR%\Scripts\asr-cli.exe transcribe "audio.mp3" --model small
echo.
echo Model cache location:
echo   %%USERPROFILE%%\.cache\huggingface\hub
echo.
echo NOTE: If you see "MKL malloc" errors, use --model tiny instead of --model small
echo.
endlocal