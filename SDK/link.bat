@echo off
FOR %%F IN (..\..\gamehook\SDK\*.h) DO (
  mklink %%~nF%%~xF %%F
)
mklink gamehook.lib ..\..\gamehook\lib64\gamehook.lib
