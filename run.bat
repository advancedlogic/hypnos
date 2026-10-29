@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================================
REM  Hypnos process manager (Windows)
REM
REM    run.bat [start|stop|restart|status] [backend|frontend|all]
REM
REM    run.bat                  -> start all
REM    run.bat restart backend  -> rebuild + restart only the Rust backend
REM    run.bat stop             -> stop backend and frontend
REM
REM  Backend : Rust server on http://localhost:8080  (cargo run, own console window)
REM  Frontend: Vite dev server on http://localhost:5173 (npm run dev, own console window)
REM  PIDs are kept in .run\*.pid so stop/restart can kill the whole process tree;
REM  the listening port is used as a fallback when no PID file exists.
REM ============================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "RUN_DIR=%ROOT%\.run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

set "BACKEND_PORT=8080"
set "FRONTEND_PORT=5173"
set "BACKEND_CONFIG=hypnos.example.toml"

REM Use the Windows tools explicitly: Git for Windows puts a Unix `find` on PATH.
set "SYS32=%SystemRoot%\System32"

REM Prefer the rustup-managed toolchain: the native engines need rustc >= 1.92 and a standalone
REM "Rust stable MSVC 1.91" install may come first on PATH.
REM `rustup run stable cargo` pins rustc as well: calling cargo.exe by path still resolves `rustc`
REM from PATH, which is the standalone 1.91 install on this machine.
REM Put the rustup toolchain first on PATH as well: `rustup run` alone does not outrank a standalone Rust install here.
if exist "%USERPROFILE%\.rustup	oolchains\stable-x86_64-pc-windows-msvcin" set "PATH=%USERPROFILE%\.rustup	oolchains\stable-x86_64-pc-windows-msvcin;%USERPROFILE%\.cargoin;%PATH%"
set "CARGO_CMD=cargo"
if exist "%USERPROFILE%\.cargo\bin\rustup.exe" set "CARGO_CMD="%USERPROFILE%\.cargo\bin\rustup.exe" run stable cargo"

set "ACTION=%~1"
set "TARGET=%~2"
if "%ACTION%"=="" set "ACTION=start"
if "%TARGET%"=="" set "TARGET=all"
if /i "%ACTION%"=="help" goto :usage
if /i "%ACTION%"=="-h" goto :usage
if /i "%ACTION%"=="--help" goto :usage

if /i not "%TARGET%"=="all" if /i not "%TARGET%"=="backend" if /i not "%TARGET%"=="frontend" (
    echo [Hypnos] Unknown target "%TARGET%".
    goto :usage
)

if /i "%ACTION%"=="start"   goto :do_start
if /i "%ACTION%"=="stop"    goto :do_stop
if /i "%ACTION%"=="restart" goto :do_restart
if /i "%ACTION%"=="status"  goto :do_status
echo [Hypnos] Unknown action "%ACTION%".
goto :usage

REM ----------------------------------------------------------------------------
:do_start
if /i "%TARGET%"=="all"      call :start_backend & call :start_frontend
if /i "%TARGET%"=="backend"  call :start_backend
if /i "%TARGET%"=="frontend" call :start_frontend
goto :end

:do_stop
if /i "%TARGET%"=="all"      call :stop_one frontend %FRONTEND_PORT% & call :stop_one backend %BACKEND_PORT%
if /i "%TARGET%"=="backend"  call :stop_one backend %BACKEND_PORT%
if /i "%TARGET%"=="frontend" call :stop_one frontend %FRONTEND_PORT%
goto :end

:do_restart
if /i "%TARGET%"=="all"      call :stop_one frontend %FRONTEND_PORT% & call :stop_one backend %BACKEND_PORT% & call :start_backend & call :start_frontend
if /i "%TARGET%"=="backend"  call :stop_one backend %BACKEND_PORT% & call :start_backend
if /i "%TARGET%"=="frontend" call :stop_one frontend %FRONTEND_PORT% & call :start_frontend
goto :end

:do_status
if /i "%TARGET%"=="all"      call :status_one backend %BACKEND_PORT% & call :status_one frontend %FRONTEND_PORT%
if /i "%TARGET%"=="backend"  call :status_one backend %BACKEND_PORT%
if /i "%TARGET%"=="frontend" call :status_one frontend %FRONTEND_PORT%
goto :end

REM ----------------------------------------------------------------------------
REM start_backend: cargo run in its own console window; the window's cmd.exe PID is saved.
:start_backend
call :is_running backend %BACKEND_PORT%
if "!RUNNING!"=="1" (
    echo [Hypnos] backend already running ^(pid !FOUND_PID!, port %BACKEND_PORT%^).
    exit /b 0
)
> "%RUN_DIR%\backend.cmd" (
    echo @echo off
    echo title Hypnos Backend ^(http://localhost:%BACKEND_PORT%^)
    echo cd /d "%ROOT%\backend"
    echo %CARGO_CMD% run --bin hypnos -- %BACKEND_CONFIG%
    echo echo.
    echo echo [Hypnos] backend exited with code %%errorlevel%%. Press any key to close.
    echo pause ^>nul
)
call :launch backend "%RUN_DIR%\backend.cmd"
if defined NEW_PID echo [Hypnos] backend starting ^(pid !NEW_PID!^) -^> http://localhost:%BACKEND_PORT%  ^(first run compiles the native engines; this takes a while^)
exit /b 0

REM start_frontend: npm run dev (Vite) in its own console window.
:start_frontend
call :is_running frontend %FRONTEND_PORT%
if "!RUNNING!"=="1" (
    echo [Hypnos] frontend already running ^(pid !FOUND_PID!, port %FRONTEND_PORT%^).
    exit /b 0
)
if not exist "%ROOT%\frontend\node_modules" (
    echo [Hypnos] Installing frontend dependencies...
    call npm --prefix "%ROOT%\frontend" install
    if errorlevel 1 (
        echo [Hypnos] Failed to install frontend dependencies.
        exit /b 1
    )
)
> "%RUN_DIR%\frontend.cmd" (
    echo @echo off
    echo title Hypnos Frontend ^(http://localhost:%FRONTEND_PORT%^)
    echo cd /d "%ROOT%\frontend"
    echo call npm run dev -- --port %FRONTEND_PORT% --strictPort
    echo echo.
    echo echo [Hypnos] frontend exited with code %%errorlevel%%. Press any key to close.
    echo pause ^>nul
)
call :launch frontend "%RUN_DIR%\frontend.cmd"
if defined NEW_PID echo [Hypnos] frontend starting ^(pid !NEW_PID!^) -^> http://localhost:%FRONTEND_PORT%
exit /b 0

REM launch <name> <launcher.cmd>: opens a new console window running the launcher and records its PID.
REM The launcher path travels through an environment variable so no quoting survives three shells.
:launch
set "NEW_PID="
set "HYPNOS_LAUNCHER=%~2"
for /f "usebackq delims=" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', ([char]34 + $env:HYPNOS_LAUNCHER + [char]34)) -PassThru).Id"`) do set "NEW_PID=%%p"
if defined NEW_PID (
    > "%RUN_DIR%\%~1.pid" echo !NEW_PID!
) else (
    echo [Hypnos] Failed to launch %~1.
)
exit /b 0

REM ----------------------------------------------------------------------------
REM stop_one <name> <port>: kill the recorded process tree, then anything still listening on the port.
:stop_one
set "STOPPED=0"
if exist "%RUN_DIR%\%~1.pid" (
    set "OLD_PID="
    set /p OLD_PID=<"%RUN_DIR%\%~1.pid"
    if defined OLD_PID (
        %SYS32%\tasklist.exe /FI "PID eq !OLD_PID!" /NH 2>nul | %SYS32%\findstr.exe /C:" !OLD_PID! " >nul && (
            %SYS32%\taskkill.exe /PID !OLD_PID! /T /F >nul 2>&1
            set "STOPPED=1"
        )
    )
    del /q "%RUN_DIR%\%~1.pid" >nul 2>&1
)
for /f "tokens=5" %%a in ('%SYS32%\netstat.exe -ano ^| %SYS32%\findstr.exe /R /C:":%~2 .*LISTENING"') do (
    if not "%%a"=="0" (
        %SYS32%\taskkill.exe /PID %%a /T /F >nul 2>&1
        set "STOPPED=1"
    )
)
if /i "%~1"=="backend" (
    REM The server binary may outlive its console if cargo was interrupted mid-run.
    %SYS32%\tasklist.exe /FI "IMAGENAME eq hypnos.exe" /NH 2>nul | %SYS32%\findstr.exe /I /C:"hypnos.exe" >nul && (
        %SYS32%\taskkill.exe /IM hypnos.exe /F >nul 2>&1
        set "STOPPED=1"
    )
)
if "!STOPPED!"=="1" (
    REM Give the OS a moment to release the port so a restart with --strictPort succeeds.
    for /l %%i in (1,1,10) do (
        %SYS32%\netstat.exe -ano | %SYS32%\findstr.exe /R /C:":%~2 .*LISTENING" >nul || goto :stop_one_done
        %SYS32%\timeout.exe /t 1 /nobreak >nul
    )
)
:stop_one_done
if "!STOPPED!"=="1" (
    echo [Hypnos] %~1 stopped.
) else (
    echo [Hypnos] %~1 was not running.
)
exit /b 0

REM ----------------------------------------------------------------------------
REM is_running <name> <port>: sets RUNNING=1 and FOUND_PID when the port is bound (preferred signal)
REM or the recorded PID is still alive.
:is_running
set "RUNNING=0"
set "FOUND_PID="
for /f "tokens=5" %%a in ('%SYS32%\netstat.exe -ano ^| %SYS32%\findstr.exe /R /C:":%~2 .*LISTENING"') do (
    if not "%%a"=="0" (
        set "RUNNING=1"
        set "FOUND_PID=%%a"
    )
)
if "!RUNNING!"=="0" if exist "%RUN_DIR%\%~1.pid" (
    set "FOUND_PID="
    set /p FOUND_PID=<"%RUN_DIR%\%~1.pid"
    if defined FOUND_PID (
        %SYS32%\tasklist.exe /FI "PID eq !FOUND_PID!" /NH 2>nul | %SYS32%\findstr.exe /C:" !FOUND_PID! " >nul && set "RUNNING=1"
    )
)
exit /b 0

:status_one
call :is_running %~1 %~2
if "!RUNNING!"=="1" (
    echo [Hypnos] %~1: RUNNING  ^(pid !FOUND_PID!, http://localhost:%~2^)
) else (
    echo [Hypnos] %~1: stopped  ^(port %~2 free^)
)
exit /b 0

REM ----------------------------------------------------------------------------
:usage
echo.
echo   Usage: run.bat [start^|stop^|restart^|status] [backend^|frontend^|all]
echo.
echo     run.bat                    start backend + frontend ^(default^)
echo     run.bat start backend      start only the Rust backend ^(port %BACKEND_PORT%^)
echo     run.bat restart frontend   restart only the Vite dev server ^(port %FRONTEND_PORT%^)
echo     run.bat stop               stop both
echo     run.bat status             show what is running
echo.
exit /b 1

:end
endlocal
exit /b 0
