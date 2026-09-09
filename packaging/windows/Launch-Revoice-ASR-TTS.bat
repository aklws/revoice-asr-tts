@echo off
setlocal EnableExtensions
set "APP_ROOT=%~dp0"
if "%APP_ROOT:~-1%"=="\" set "APP_ROOT=%APP_ROOT:~0,-1%"
set "UV_PROJECT_ENVIRONMENT=%APP_ROOT%\.venv"
set "VENV_PYTHON=%UV_PROJECT_ENVIRONMENT%\Scripts\python.exe"
set "MAIN_PY=%APP_ROOT%\main.py"
set "PYTHONPATH=%APP_ROOT%"
set "PATH=%APP_ROOT%\bin\ffmpeg\bin;%APP_ROOT%\bin\ffmpeg;%PATH%"
set "PYTHONUTF8=1"
set "BOOTSTRAP_PYTHON="

if exist "%VENV_PYTHON%" goto run_with_venv

rem Prefer the uv.exe bundled next to this launcher. It is always present
rem in the packaged app, so we never depend on PATH or a system install.
if exist "%APP_ROOT%\uv.exe" (
    set "PATH=%APP_ROOT%;%PATH%"
    goto sync_with_uv
)

where uv >nul 2>nul
if not errorlevel 1 goto sync_with_uv

rem uv is not on PATH. The official uv installer stores uv.exe in
rem "%USERPROFILE%\.local\bin" but often leaves it off PATH, which made the
rem launcher fail right away ("flash exit") even though uv is installed.
rem Reuse that binary directly before falling back to installing uv.
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    goto sync_with_uv
)

call :find_bootstrap_python
if not defined BOOTSTRAP_PYTHON (
    echo.
    echo [ERROR] uv was not found, and no compatible Python runtime was detected.
    echo.
    echo Install one of the following, then run this launcher again:
    echo   - uv  (recommended) : https://docs.astral.sh/uv/getting-started/installation/
    echo   - Python 3.13       : https://www.python.org/downloads/
    echo.
    goto pause_and_exit
)

echo.
echo uv was not found. Bootstrapping a local environment...
pushd "%APP_ROOT%"
call :ensure_uv
if errorlevel 1 (
    popd
    goto pause_and_exit
)
%BOOTSTRAP_PYTHON% -m uv sync --frozen
if errorlevel 1 (
    echo.
    echo [ERROR] uv sync failed. Check your network connection, then try again.
    popd
    goto pause_and_exit
)
if not exist "%VENV_PYTHON%" (
    echo.
    echo [ERROR] Local environment creation failed: %VENV_PYTHON%
    popd
    goto pause_and_exit
)
"%VENV_PYTHON%" "%MAIN_PY%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
goto exit_app

:sync_with_uv
pushd "%APP_ROOT%"
uv sync --frozen
if errorlevel 1 (
    echo.
    echo [ERROR] uv sync failed. Check your network connection, then try again.
    popd
    goto pause_and_exit
)
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%MAIN_PY%" %*
) else (
    uv run python "%MAIN_PY%" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
popd
goto exit_app

:run_with_venv
pushd "%APP_ROOT%"
"%VENV_PYTHON%" "%MAIN_PY%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
goto exit_app

:find_bootstrap_python
py -3.13 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=py -3.13"
    exit /b 0
)
py -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=py"
    exit /b 0
)
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "BOOTSTRAP_PYTHON=python"
)
exit /b 0

:ensure_uv
%BOOTSTRAP_PYTHON% -m uv --version >nul 2>nul
if not errorlevel 1 exit /b 0
echo.
echo Installing uv for the current user...
%BOOTSTRAP_PYTHON% -m pip install --user --disable-pip-version-check uv
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install uv automatically.
    echo Install uv manually, then run this launcher again:
    echo https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
%BOOTSTRAP_PYTHON% -m uv --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] uv installation finished, but the launcher still could not start uv.
    echo Install uv manually, then run this launcher again:
    echo https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
exit /b 0

rem Keep the console window open on failure so the user can read the error
rem instead of the window closing ("flash exit") immediately.
:pause_and_exit
echo.
pause
exit /b 1

:exit_app
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Revoice exited with code %EXIT_CODE%.
    echo.
    pause
)
exit /b %EXIT_CODE%
