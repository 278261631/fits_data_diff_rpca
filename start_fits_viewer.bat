@echo off
setlocal

cd /d "%~dp0"

python fits_viewer.py

if errorlevel 1 (
    echo.
    echo 启动失败。请确认已安装依赖：
    echo   pip install PySide6 astropy numpy
    echo.
    pause
)
