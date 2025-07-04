@echo off
echo Setting up Python ML Environment...
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv ml_env
if errorlevel 1 (
    echo Error creating virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment and install packages
echo.
echo Activating virtual environment and installing packages...
call ml_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo Error installing packages
    pause
    exit /b 1
)

echo.
echo ✅ Setup complete!
echo.
echo To activate the environment in the future, run:
echo   ml_env\Scripts\activate
echo.
echo To start Jupyter Lab, run:
echo   jupyter lab
echo.
pause
