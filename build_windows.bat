@echo off
REM Build script for 2048 AI on Windows
REM Adjust paths below to match your system

SET LIBTORCH_PATH=C:\libtorch
SET VCPKG_PATH=C:\vcpkg

echo =============================================
echo 2048 AI Neural Network - Windows Build Script
echo =============================================
echo.

REM Check if build directory exists
if not exist build mkdir build
cd build

echo Configuring with CMake...
cmake .. -DCMAKE_PREFIX_PATH="%LIBTORCH_PATH%;%VCPKG_PATH%\installed\x64-windows" ^
         -DCMAKE_TOOLCHAIN_FILE="%VCPKG_PATH%\scripts\buildsystems\vcpkg.cmake" ^
         -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CMake configuration failed!
    echo Make sure LIBTORCH_PATH and VCPKG_PATH are set correctly.
    pause
    exit /b 1
)

echo.
echo Building...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo =============================================
echo Build successful!
echo Executable: build\Release\2048_ai.exe
echo =============================================

cd ..
pause
