@echo off
title Push to GitHub
setlocal enabledelayedexpansion

echo ======================================================================
echo                 Aegis AI - GitHub Push Utility
echo ======================================================================
echo.

:: Check git installation
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

:: Get repo URL from argument or user prompt
set "REPO_URL=%~1"
if "%REPO_URL%"=="" (
    set /p "REPO_URL=Enter your GitHub Repository URL (e.g. https://github.com/user/repo.git): "
)

if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL provided. Aborting.
    pause
    exit /b 1
)

echo.
echo [*] Staging files...
git add .

echo [*] Committing changes...
git commit -m "Initial commit: Aegis AI Threat Detection & SOC Dashboard"

echo [*] Setting primary branch to main...
git branch -M main

echo [*] Configuring remote origin...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo [*] Pushing to GitHub (main branch)...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ======================================================================
    echo   [SUCCESS] Project successfully pushed to GitHub!
    echo ======================================================================
) else (
    echo.
    echo [!] If push was rejected, check that the remote repo exists and you have write permissions.
    echo     You can retry with: git push -u origin main --force
)

echo.
pause
