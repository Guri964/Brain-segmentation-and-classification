@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Step 4: Check if Python 3.10 is installed
echo Checking for Python 3.10 installation...
python --version 2>nul | findstr "3.10" >nul
if %errorlevel% neq 0 (
    echo Python 3.10 not found. Downloading and installing Python 3.10...
    curl -o python-3.10-installer.exe https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
    echo Running Python installer interactively...
    start /wait python-3.10-installer.exe
    del python-3.10-installer.exe
) else (
    echo Python 3.10 is already installed.
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Install requirements
echo Installing dependencies...
pip install --upgrade pip
pip install -r req.txt

echo Installation complete!
pause
