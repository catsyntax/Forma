@echo off
rem Forma for Windows. Double-click to start.
rem The first start installs a private Python and all libraries into the ".venv" folder here (5-10 minutes)
rem and adds a Forma shortcut to your Start menu. Nothing else on your PC is changed.
setlocal
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "CHECK=import PySide6, pyvista, rhino3dm, OCP, trimesh, ezdxf, networkx"

if not exist ".venv\Scripts\python.exe" goto setup
".venv\Scripts\python.exe" -c "%CHECK%" >nul 2>&1 || goto setup
fc /b requirements.txt .venv\.installed >nul 2>&1 || goto setup
goto run

:setup
echo == Setting up Forma (first start only) ==
where uv >nul 2>&1
if errorlevel 1 (
  echo -- Installing the 'uv' Python manager...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto fail
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.12 .venv || goto fail
)
echo -- Installing libraries (Qt, VTK, OpenCascade...). About 1 GB the first time.
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto fail
copy /y requirements.txt .venv\.installed >nul
rem Start-menu shortcut
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Programs')+'\Forma.lnk'); $s.TargetPath='%~dp0start_windows.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()" >nul 2>&1

:run
echo Starting Forma...
".venv\Scripts\python.exe" -m forma %*
goto :eof

:fail
echo.
echo Setup failed. Check your internet connection and try again.
pause
