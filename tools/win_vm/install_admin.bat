@echo off
chcp 65001 >nul
echo Установка задач очистки временной папки FineReader...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Tools\install_cleanup_task.ps1'"
echo.
pause
