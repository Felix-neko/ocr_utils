@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Tools\status_cleanup.ps1'"
echo.
pause
