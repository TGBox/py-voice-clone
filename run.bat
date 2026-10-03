@echo off
title VoiceClone AI Studio
cd /d "%~dp0"

echo ========================================================
echo        VoiceClone AI Studio - XTTS-v2 Neural Studio
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtuelle Python-Umgebung nicht gefunden!
    echo Bitte fuehren Sie zuerst die Einrichtung durch.
    pause
    exit /b 1
)

echo [*] Starte VoiceClone AI Studio mit NVIDIA RTX Beschleunigung...
start "" ".venv\Scripts\pythonw.exe" main.py
exit
