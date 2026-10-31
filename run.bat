@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================================
REM  Hypnos process manager (Windows)
REM
REM    run.bat [build|start|stop|restart|status] [backend|frontend|desktop|all]
REM    run.bat models [all|<model id>]   download model files into backend\models\
REM
REM    run.bat build            -> compile the backend (cargo build --release) + npm install
REM    run.bat                  -> start all (never compiles)
REM    run.bat restart backend  -> stop + start the already-built backend binary
REM    run.bat stop             -> stop backend and frontend
REM    run.bat build desktop    -> npm install + tauri build of the desktop app (not part of `all`)
REM
REM  Build and start are separate steps: `start` launches the compiled executable
REM  backend\target\<profile>\hypnos.exe and refuses to run when it is missing, so a
REM  stale or in-progress cargo build can never masquerade as a running server.
REM  Set HYPNOS_PROFILE=debug to build/start the dev profile instead of release, and
REM  HYPNOS_CONFIG=<file> (relative to backend\) to start the backend with your own config.
REM
REM  Backend : Rust server on http://localhost:9999  (hypnos.exe, own console window)
REM  Frontend: Vite dev server on http://localhost:5173 (npm run dev, own console window)
REM  Desktop : Tauri app (desktop\src-tauri\target\<profile>\hypnos-desktop.exe), only when named
REM  PIDs are kept in .run\*.pid so stop/restart can kill the whole process tree;
REM  the listening port is used as a fallback when no PID file exists.
REM ============================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "RUN_DIR=%ROOT%\.run"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

set "BACKEND_PORT=9999"
set "FRONTEND_PORT=5173"
set "BACKEND_CONFIG=hypnos.example.toml"
if defined HYPNOS_CONFIG set "BACKEND_CONFIG=%HYPNOS_CONFIG%"

REM Cargo profile for the backend binary: release (default) or debug (`cargo build` dev profile).
if not defined HYPNOS_PROFILE set "HYPNOS_PROFILE=release"
if /i "%HYPNOS_PROFILE%"=="dev" set "HYPNOS_PROFILE=debug"
set "CARGO_PROFILE_FLAG=--profile %HYPNOS_PROFILE%"
if /i "%HYPNOS_PROFILE%"=="release" set "CARGO_PROFILE_FLAG=--release"
if /i "%HYPNOS_PROFILE%"=="debug"   set "CARGO_PROFILE_FLAG="
set "BACKEND_EXE=%ROOT%\backend\target\%HYPNOS_PROFILE%\hypnos.exe"
REM The desktop app follows the same profile: `tauri build` (release) or `tauri build --debug`.
set "DESKTOP_EXE=%ROOT%\desktop\src-tauri\target\%HYPNOS_PROFILE%\hypnos-desktop.exe"
set "DESKTOP_BUILD_FLAG="
if /i "%HYPNOS_PROFILE%"=="debug" set "DESKTOP_BUILD_FLAG=--debug"

REM Use the Windows tools explicitly: Git for Windows puts a Unix `find` on PATH.
set "SYS32=%SystemRoot%\System32"

REM Prefer the rustup-managed toolchain: the native engines need rustc >= 1.92 and a standalone
REM "Rust stable MSVC 1.91" install may come first on PATH.
REM `rustup run stable cargo` pins rustc as well: calling cargo.exe by path still resolves `rustc`
REM from PATH, which is the standalone 1.91 install on this machine.
REM Put the rustup toolchain first on PATH as well: `rustup run` alone does not outrank a standalone Rust install here.
if exist "%USERPROFILE%\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin" set "PATH=%USERPROFILE%\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin;%USERPROFILE%\.cargo\bin;%PATH%"
set "CARGO_CMD=cargo"
if exist "%USERPROFILE%\.cargo\bin\rustup.exe" set "CARGO_CMD="%USERPROFILE%\.cargo\bin\rustup.exe" run stable cargo"

set "ACTION=%~1"
set "TARGET=%~2"
if "%ACTION%"=="" set "ACTION=start"
if "%TARGET%"=="" set "TARGET=all"
if /i "%ACTION%"=="help" goto :usage
if /i "%ACTION%"=="-h" goto :usage
if /i "%ACTION%"=="--help" goto :usage
if /i "%ACTION%"=="models" goto :do_models

if /i not "%TARGET%"=="all" if /i not "%TARGET%"=="backend" if /i not "%TARGET%"=="frontend" if /i not "%TARGET%"=="desktop" (
    echo [Hypnos] Unknown target "%TARGET%".
    goto :usage
)

REM The desktop app is never part of `all`: it needs the Tauri prerequisites and opens a window.
if /i "%TARGET%"=="desktop" (
    if /i "%ACTION%"=="build"   call :build_desktop & set "RC=!errorlevel!" & goto :end_rc
    if /i "%ACTION%"=="start"   call :start_desktop & set "RC=!errorlevel!" & goto :end_rc
    if /i "%ACTION%"=="stop"    call :stop_desktop & goto :end
    if /i "%ACTION%"=="restart" call :stop_desktop & call :start_desktop & set "RC=!errorlevel!" & goto :end_rc
    if /i "%ACTION%"=="status"  call :status_desktop & goto :end
    echo [Hypnos] Unknown action "%ACTION%".
    goto :usage
)

if /i "%ACTION%"=="build"   goto :do_build
if /i "%ACTION%"=="start"   goto :do_start
if /i "%ACTION%"=="stop"    goto :do_stop
if /i "%ACTION%"=="restart" goto :do_restart
if /i "%ACTION%"=="status"  goto :do_status
echo [Hypnos] Unknown action "%ACTION%".
goto :usage

REM ----------------------------------------------------------------------------
:do_models
REM backend\models.manifest lists the files; existing ones are kept, interrupted ones resume.
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\backend\scripts\fetch-models.ps1" %TARGET%
set "RC=%ERRORLEVEL%"
goto :end_rc

:do_build
set "RC=0"
if /i not "%TARGET%"=="frontend" call :build_backend
if errorlevel 1 set "RC=1"
if /i not "%TARGET%"=="backend"  call :build_frontend
if errorlevel 1 set "RC=1"
goto :end_rc

:do_start
set "RC=0"
if /i not "%TARGET%"=="frontend" call :start_backend
if errorlevel 1 set "RC=1"
if /i not "%TARGET%"=="backend"  call :start_frontend
if errorlevel 1 set "RC=1"
goto :end_rc

:do_stop
if /i not "%TARGET%"=="backend"  call :stop_one frontend %FRONTEND_PORT%
if /i not "%TARGET%"=="frontend" call :stop_one backend %BACKEND_PORT%
goto :end

:do_restart
set "RC=0"
if /i not "%TARGET%"=="backend"  call :stop_one frontend %FRONTEND_PORT%
if /i not "%TARGET%"=="frontend" call :stop_one backend %BACKEND_PORT%
if /i not "%TARGET%"=="frontend" call :start_backend
if errorlevel 1 set "RC=1"
if /i not "%TARGET%"=="backend"  call :start_frontend
if errorlevel 1 set "RC=1"
goto :end_rc

:do_status
if /i not "%TARGET%"=="frontend" call :status_one backend %BACKEND_PORT%
if /i not "%TARGET%"=="backend"  call :status_one frontend %FRONTEND_PORT%
goto :end

REM ----------------------------------------------------------------------------
REM build_backend: cargo build in the current console (progress is visible, exit code propagates).
REM The vendored engine build scripts copy stable-diffusion.dll next to the produced binary.
:build_backend
echo [Hypnos] Building backend ^(%HYPNOS_PROFILE% profile^) -^> %BACKEND_EXE%
echo          The first build compiles the native engines ^(llama.cpp, stable-diffusion.cpp, espeak-ng^); this takes a while.
call :ensure_msvc_env
pushd "%ROOT%\backend"
REM --features docling-cuda: ONNX Runtime's CUDA provider for docling's device = "gpu", as in the
REM release archives (prebuilt by ort; needs the CUDA 13 runtime and cuDNN 9 only when used).
%CARGO_CMD% build %CARGO_PROFILE_FLAG% --features docling-cuda --bin hypnos
set "BUILD_RC=!errorlevel!"
popd
if not "!BUILD_RC!"=="0" (
    echo [Hypnos] backend build FAILED ^(cargo exit code !BUILD_RC!^).
    exit /b 1
)
if not exist "%BACKEND_EXE%" (
    echo [Hypnos] cargo reported success but %BACKEND_EXE% does not exist.
    exit /b 1
)
echo [Hypnos] backend built: %BACKEND_EXE%
exit /b 0

REM ensure_msvc_env: the engines' CMake projects need the MSVC toolchain, including rc.exe (the
REM Windows resource compiler), on PATH. A plain terminal does not have it; a "Developer Command
REM Prompt" does. When rc.exe is missing, load vcvars64.bat from the newest Visual Studio with the
REM C++ workload so `run.bat build` works from any console.
:ensure_msvc_env
where rc.exe >nul 2>&1 && exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
set "VCVARS_LATEST="
if exist "%VSWHERE%" (
    REM Prefer the install that ships an MSVC 14.4x toolset: CUDA 13 only supports that host
    REM compiler family (see README), and .cargo\config.toml pins CUDAHOSTCXX to it. Fall back to
    REM the newest install otherwise. vswhere lists installs newest first.
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if not defined VCVARS_LATEST set "VCVARS_LATEST=%%i\VC\Auxiliary\Build\vcvars64.bat"
        if not defined VCVARS if exist "%%i\VC\Tools\MSVC\14.4*" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS set "VCVARS=%VCVARS_LATEST%"
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS (
    echo [Hypnos] rc.exe is not on PATH and no Visual Studio C++ toolset was found; run this from a Developer Command Prompt.
    exit /b 0
)
if not exist "%VCVARS%" (
    echo [Hypnos] rc.exe is not on PATH and %VCVARS% does not exist; run this from a Developer Command Prompt.
    exit /b 0
)
echo [Hypnos] Loading the MSVC developer environment ^(%VCVARS%^)
REM vcvars itself shells out to vswhere.exe by bare name; make sure it can find it.
if exist "%VSWHERE%" set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VCVARS%" >nul
where rc.exe >nul 2>&1 || echo [Hypnos] WARNING: rc.exe still not on PATH after vcvars64.bat; the Windows SDK may be missing from that Visual Studio install.
exit /b 0

REM build_frontend: install/refresh node dependencies. The Vite dev server bundles on the fly.
:build_frontend
echo [Hypnos] Installing frontend dependencies ^(npm install^)...
pushd "%ROOT%\frontend"
call npm install
set "BUILD_RC=!errorlevel!"
popd
if not "!BUILD_RC!"=="0" (
    echo [Hypnos] frontend dependency install FAILED ^(npm exit code !BUILD_RC!^).
    exit /b 1
)
echo [Hypnos] frontend dependencies ready.
exit /b 0

REM build_desktop: npm install + `tauri build --no-bundle` (the app binary; release.yml makes the
REM installers). Tauri compiles the Rust side with the cargo first on PATH (rustup's, set above).
:build_desktop
echo [Hypnos] Building the desktop app ^(%HYPNOS_PROFILE% profile^) -^> %DESKTOP_EXE%
call :ensure_msvc_env
pushd "%ROOT%\desktop"
call npm install
set "BUILD_RC=!errorlevel!"
if "!BUILD_RC!"=="0" (
    call npx tauri build --no-bundle %DESKTOP_BUILD_FLAG%
    set "BUILD_RC=!errorlevel!"
)
popd
if not "!BUILD_RC!"=="0" (
    echo [Hypnos] desktop build FAILED ^(exit code !BUILD_RC!^).
    exit /b 1
)
echo [Hypnos] desktop app built: %DESKTOP_EXE%
exit /b 0

REM ----------------------------------------------------------------------------
REM start_backend: run the compiled executable in its own console window; the window's cmd.exe PID is saved.
REM Never compiles: a missing binary is an error that points at `run.bat build backend`.
:start_backend
call :is_running backend %BACKEND_PORT%
if "!RUNNING!"=="1" (
    echo [Hypnos] backend already running ^(pid !FOUND_PID!, port %BACKEND_PORT%^).
    exit /b 0
)
if not exist "%BACKEND_EXE%" (
    echo [Hypnos] backend binary not found: %BACKEND_EXE%
    echo          Run `run.bat build backend` first ^(or set HYPNOS_PROFILE to the profile you built^).
    exit /b 1
)
> "%RUN_DIR%\backend.cmd" (
    echo @echo off
    echo title Hypnos Backend ^(http://localhost:%BACKEND_PORT%^)
    echo cd /d "%ROOT%\backend"
    echo "%BACKEND_EXE%" %BACKEND_CONFIG%
    echo echo.
    echo echo [Hypnos] backend exited with code %%errorlevel%%. Press any key to close.
    echo pause ^>nul
)
call :launch backend "%RUN_DIR%\backend.cmd"
if not defined NEW_PID exit /b 1
echo [Hypnos] backend starting ^(pid !NEW_PID!, %HYPNOS_PROFILE% build^) -^> http://localhost:%BACKEND_PORT%
exit /b 0

REM start_frontend: npm run dev (Vite) in its own console window.
:start_frontend
call :is_running frontend %FRONTEND_PORT%
if "!RUNNING!"=="1" (
    echo [Hypnos] frontend already running ^(pid !FOUND_PID!, port %FRONTEND_PORT%^).
    exit /b 0
)
if not exist "%ROOT%\frontend\node_modules" (
    echo [Hypnos] frontend dependencies not installed: %ROOT%\frontend\node_modules
    echo          Run `run.bat build frontend` first.
    exit /b 1
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
if not defined NEW_PID exit /b 1
echo [Hypnos] frontend starting ^(pid !NEW_PID!^) -^> http://localhost:%FRONTEND_PORT%
exit /b 0

REM start_desktop: open the built app (a GUI executable, no console window); its PID is saved.
:start_desktop
%SYS32%\tasklist.exe /FI "IMAGENAME eq hypnos-desktop.exe" /NH 2>nul | %SYS32%\findstr.exe /I /C:"hypnos-desktop.exe" >nul && (
    echo [Hypnos] desktop app already running.
    exit /b 0
)
if not exist "%DESKTOP_EXE%" (
    echo [Hypnos] desktop app not built: %DESKTOP_EXE%
    echo          Run `run.bat build desktop` first.
    exit /b 1
)
set "NEW_PID="
set "HYPNOS_LAUNCHER=%DESKTOP_EXE%"
for /f "usebackq delims=" %%p in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(Start-Process -FilePath $env:HYPNOS_LAUNCHER -WorkingDirectory (Split-Path $env:HYPNOS_LAUNCHER) -PassThru).Id"`) do set "NEW_PID=%%p"
if not defined NEW_PID (
    echo [Hypnos] Failed to launch the desktop app.
    exit /b 1
)
> "%RUN_DIR%\desktop.pid" echo !NEW_PID!
echo [Hypnos] desktop app RUNNING ^(pid !NEW_PID!^)
exit /b 0

:stop_desktop
del /q "%RUN_DIR%\desktop.pid" >nul 2>&1
%SYS32%\tasklist.exe /FI "IMAGENAME eq hypnos-desktop.exe" /NH 2>nul | %SYS32%\findstr.exe /I /C:"hypnos-desktop.exe" >nul && (
    %SYS32%\taskkill.exe /IM hypnos-desktop.exe /F >nul 2>&1
    echo [Hypnos] desktop app stopped.
    exit /b 0
)
echo [Hypnos] desktop app was not running.
exit /b 0

:status_desktop
%SYS32%\tasklist.exe /FI "IMAGENAME eq hypnos-desktop.exe" /NH 2>nul | %SYS32%\findstr.exe /I /C:"hypnos-desktop.exe" >nul && (
    echo [Hypnos] desktop: RUNNING
) || (
    echo [Hypnos] desktop: stopped
)
if exist "%DESKTOP_EXE%" (
    echo [Hypnos] desktop binary: %DESKTOP_EXE%
) else (
    echo [Hypnos] desktop binary: not built ^(run.bat build desktop^)
)
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
    REM The server binary may outlive its console window.
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
if /i "%~1"=="backend" (
    if exist "%BACKEND_EXE%" (
        echo [Hypnos] backend binary: %BACKEND_EXE%
    ) else (
        echo [Hypnos] backend binary: not built ^(run.bat build backend^)
    )
)
exit /b 0

REM ----------------------------------------------------------------------------
:usage
echo.
echo   Usage: run.bat [build^|start^|stop^|restart^|status] [backend^|frontend^|desktop^|all]
echo.
echo     run.bat build              cargo build --release the backend + npm install the frontend
echo     run.bat build backend      compile only the Rust server ^(backend\target\%HYPNOS_PROFILE%\hypnos.exe^)
echo     run.bat                    start backend + frontend from the built artifacts ^(default^)
echo     run.bat start backend      start only the Rust backend binary ^(port %BACKEND_PORT%^)
echo     run.bat restart frontend   restart only the Vite dev server ^(port %FRONTEND_PORT%^)
echo     run.bat stop               stop both
echo     run.bat status             show what is running and whether the backend is built
echo     run.bat models [model id]  download the model files ^(about 31 GB for all^) into backend\models\
echo     run.bat build desktop      npm install + tauri build the desktop app ^(no installers^)
echo     run.bat start desktop      open the built desktop app ^("all" leaves it out^)
echo.
echo   start/restart never compile; run `run.bat build` after changing Rust code.
echo   HYPNOS_PROFILE=debug selects the dev profile ^(backend\target\debug\hypnos.exe^).
echo   For desktop development with hot reload: cd desktop ^&^& npx tauri dev
echo.
exit /b 1

:end_rc
endlocal & exit /b %RC%

:end
endlocal
exit /b 0
