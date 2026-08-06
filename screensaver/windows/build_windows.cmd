@echo off
rem Build SmytheGlyphRain.scr with the C# compiler bundled with Windows.
rem No SDK, NuGet, or network access required. Run from any directory.

setlocal
set HERE=%~dp0
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
    echo Could not find the .NET Framework C# compiler.
    exit /b 1
)

if not exist "%HERE%..\dist" mkdir "%HERE%..\dist"

"%CSC%" /nologo /target:winexe /optimize+ /warn:4 ^
    /reference:System.dll /reference:System.Drawing.dll ^
    /reference:System.Windows.Forms.dll ^
    /out:"%HERE%..\dist\SmytheGlyphRain.scr" ^
    "%HERE%GlyphRainSaver.cs" "%HERE%GlyphData.cs"

if errorlevel 1 exit /b 1
echo Built %HERE%..\dist\SmytheGlyphRain.scr
echo Install: right-click the .scr and choose "Install", or copy to a folder
echo and select it under Settings ^> Personalization ^> Lock screen ^> Screensaver.
endlocal
