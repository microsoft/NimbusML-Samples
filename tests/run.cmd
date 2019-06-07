@if not defined _echo @echo off
setlocal

:: Store current script directory before %~dp0 gets affected by another process later.
set __currentScriptDir=%~dp0

:: Default configuration
set DependenciesDir=%__currentScriptDir%dependencies\
set PythonUrl=https://pythonpkgdeps.blob.core.windows.net/python/python-3.6.5-mohoov-amd64.zip
set PythonRoot=%DependenciesDir%Python3.6
set PythonExe=%PythonRoot%\python.exe
set GraphvizUrl=https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.zip
set GraphvizRoot=%DependenciesDir%Graphviz
set AlexnetUrl=http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb
set AlexnetFile=%__currentScriptDir%..\samples\alexnet_frozen.pb

if [%1] == [] goto :Main
goto :Usage

:Usage
echo Usage: run.cmd
echo Description: Runs all python notebooks in the samples folder.
goto :Exit_Success


:Main
echo ########################################
echo Preparing to run all Python samples
echo ########################################
if not exist "%DependenciesDir%" (md "%DependenciesDir%")
:: Download & unzip Python
if not exist "%PythonRoot%\.done" (
    md "%PythonRoot%"
    echo Downloading python zip ... 
    powershell -command "& {$wc = New-Object System.Net.WebClient; $wc.DownloadFile('%PythonUrl%', '%DependenciesDir%python.zip');}"
    echo Extracting python zip ... 
    powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%DependenciesDir%python.zip', '%PythonRoot%'); }"
    echo.>"%PythonRoot%\.done"
    del %DependenciesDir%python.zip
)
:: Download Alexnet file
if not exist "%AlexnetFile%" (
    echo Downloading Alexnet frozen model for use in Notebook 2.3 "Image Processing" ...
    powershell -command "& {$wc = New-Object System.Net.WebClient; $wc.DownloadFile('%AlexnetUrl%', '%AlexnetFile%');}"
)
:: Download & unzip Graphviz
if not exist "%GraphvizRoot%\.done" (
    md "%GraphvizRoot%"
    echo Downloading graphviz for use in Notebook 3.1 "Visualize a Pipeline" ...
    powershell -command "& {$wc = New-Object System.Net.WebClient; $wc.DownloadFile('%GraphvizUrl%', '%DependenciesDir%graphviz.zip');}"
    echo Extracting graphviz zip ... 
    powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%DependenciesDir%graphviz.zip', '%GraphvizRoot%'); }"
    echo.>"%GraphvizRoot%\.done"
    del %DependenciesDir%graphviz.zip
)
echo Temporarily adding graphviz to path
set PATH=%PATH%;%GraphvizRoot%\release\bin


:: Install nimbusml and run all sample notebooks
set SampleDependencies=graphviz jupyter "nbconvert>=4.2.0" matplotlib requests Image
echo Pip installing nimbusml and required modules
call "%PythonExe%" -m pip install %SampleDependencies%
call "%PythonExe%" -m pip install nimbusml
:: This is a workaround for a conflict between jupyter and nbconvert. jupyter forces prompt-toolkit<2.0, and nbconvert
:: imports features at runtime from prompt-toolkit=2.0. Here we upgrade prompt-toolkit after the install of jupyter to get around that.
call "%PythonExe%" -m pip install "prompt-toolkit>=2.0"

echo.
echo #######################################################
echo Running all Python notebooks in Samples directory ...
echo #######################################################
call "%PythonExe%" %__currentScriptDir%run_samples.py
:: Retry once as there are occasional failures (that I don't currently understand) in the image tutorial notebook, cell 7
if %ERRORLEVEL% NEQ 0 call "%PythonExe%" %__currentScriptDir%run_samples.py
goto :Exit_Success


:Exit_Success
endlocal
exit /b %ERRORLEVEL%

