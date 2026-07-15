@echo off
REM ============================================================
REM  Annotation Pipeline — Virtual Environment Bootstrap
REM  Run this on any machine to recreate .venv_annotation
REM ============================================================
REM  Prerequisites:
REM    - Python 3.12 installed and on PATH  (py -3.12 --version)
REM    - CUDA 11.8 or 12.x GPU driver
REM    - Internet access for pip downloads
REM    - Grounded-SAM-2 repo cloned separately (see SETUP.md)
REM ============================================================

setlocal enabledelayedexpansion

REM Resolve the project root (parent of this script's directory)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\"
pushd "%PROJECT_ROOT%"
set "PROJECT_ROOT=%CD%"
popd

set "VENV_DIR=%PROJECT_ROOT%\.venv_annotation"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

echo.
echo === Annotation Pipeline VEnv Bootstrap ===
echo   Project root : %PROJECT_ROOT%
echo   Venv target  : %VENV_DIR%
echo.

REM --- Check Python 3.12 available ---
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 not found.
    echo Install from https://www.python.org/downloads/ ^(not the Microsoft Store version^)
    exit /b 1
)
echo [1/5] Python 3.12 found.

REM --- Create venv if it doesn't exist ---
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [2/5] Venv already exists, skipping creation.
) else (
    echo [2/5] Creating virtual environment with Python 3.12...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
    echo       Created: %VENV_DIR%
)

REM --- Upgrade pip ---
echo [3/5] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo WARNING: pip upgrade failed, continuing anyway.
)

REM --- Install PyTorch cu118 ---
echo [4/5] Installing PyTorch 2.3.1 with CUDA 11.8...
echo       ^(This is ~2.5 GB — may take several minutes^)
"%PIP_EXE%" install ^
    "torch==2.3.1+cu118" ^
    "torchvision==0.18.1+cu118" ^
    "torchaudio==2.3.1+cu118" ^
    --index-url https://download.pytorch.org/whl/cu118 ^
    --quiet
if errorlevel 1 (
    echo ERROR: PyTorch installation failed.
    echo Check your internet connection and CUDA driver version.
    exit /b 1
)
echo       PyTorch installed.

REM --- Install remaining requirements ---
echo [5/5] Installing annotation pipeline dependencies...
"%PIP_EXE%" install -r "%REQ_FILE%" --quiet
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt
    exit /b 1
)
echo       Dependencies installed.

REM --- Verify CUDA is visible ---
echo.
echo === Verifying CUDA availability ===
"%PYTHON_EXE%" -c "import torch; cuda=torch.cuda.is_available(); print('  CUDA available:', cuda); print('  PyTorch version:', torch.__version__); print('  CUDA version (torch):', torch.version.cuda if cuda else 'N/A'); name=torch.cuda.get_device_name(0) if cuda else 'N/A'; print('  GPU:', name)"
if errorlevel 1 (
    echo WARNING: CUDA check failed. The venv is installed but GPU may not be accessible.
)

echo.
echo === Done ===
echo.
echo Next steps:
echo   1. Clone Grounded-SAM-2 repo and install SAM2 + Grounding DINO
echo      ^(see scripts\annotation\SETUP.md Steps 1 and 4-6^)
echo   2. Download model weights ^(SETUP.md Steps 7-8^)
echo   3. Update GSAM2_REPO paths in scripts\annotation\auto_annotate.py
echo   4. Run: "%PYTHON_EXE%" scripts\annotation\auto_annotate.py --verify
echo.
pause
